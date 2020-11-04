import argparse
import os

import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

from models.lowdim import SimplePendulum, HarmonicOscillator
from models.integrators.implicit_midpoint import ImplicitMidpointIntegrator
from models.integrators.stormer_verlet import SeparableStormerVerletIntegrator

from nn.models import integrate
from nn.linearsymplectic import *
from nn.nonlinearsymplectic import *
from nn.symplecticloss import symplectic_mse_loss

from utils.plot2d import *
from utils.data import td_x_to_hamiltonian, time_data_list_to_numpy, save_json

class SamplingParameters:
    def __init__(self, mu, qmin=-1, qmax=+1, pmin=-1, pmax=+1):
        self.n = 10000

        self.qmin = qmin
        self.qmax = qmax

        self.pmin = pmin
        self.pmax = pmax

        self.mu = mu

class Configuration:
    def __init__(self, experiment, name, mu, t_start, t_end):
        self.experiment = experiment
        self.name = name
        self.mu = mu
        self.t_start = t_start
        self.t_end = t_end

        # cache solution
        self._td_x, self._td_Ham = self.experiment.model.solve(self.t_start, self.t_end, 
            self.experiment.num_integrator, self.mu)

        # save arguments on disk
        save_json((experiment.output_dir, name, 'args.json'), {
            'name': name,
            'mu': mu,
            't_start': t_start,
            't_end': t_end
        })

        # save exact solution on disk
        np.save(os.path.join(experiment.output_dir, name, 'exact_td_x.npy'), 
            self._td_x.all_data_to_numpy())
        np.save(os.path.join(experiment.output_dir, name, 'exact_td_Ham.npy'), 
            time_data_list_to_numpy(self._td_Ham))

    def run(self, epoch):
        td_x_surrogate = integrate(self.experiment.surrogate_model, self.mu['q0'], self.mu['p0'], 
            self.t_start, self.t_end, self.experiment.dt)
        td_Ham_surrogate = td_x_to_hamiltonian(td_x_surrogate, self.experiment.model, self.mu)

        # save data on disk
        np.save(os.path.join(self.experiment.output_dir, self.name, 'epoch{}_td_x.npy'.format(epoch)), 
            td_x_surrogate.all_data_to_numpy())
        np.save(os.path.join(self.experiment.output_dir, self.name, 'epoch{}_td_Ham.npy'.format(epoch)), 
            time_data_list_to_numpy(td_Ham_surrogate))

        # plot results
        plt = plot_Ham_sys(self._td_x, td_x_surrogate)
        self.experiment.writer.add_figure(self.name + '/PhaseSpace', plt, epoch)

        qplt = plot_q(self._td_x, td_x_surrogate)
        self.experiment.writer.add_figure(self.name + '/q', qplt, epoch)

        pplt = plot_p(self._td_x, td_x_surrogate)
        self.experiment.writer.add_figure(self.name + '/p', pplt, epoch)

        ham_plt = plot_hamiltonian(self._td_Ham, td_Ham_surrogate)
        self.experiment.writer.add_figure(self.name + '/Hamiltonian', ham_plt, epoch)

class Experiment:

    def __init__(self, args, model, surrogate_model, sampling_params: SamplingParameters):
        self.dt = args.dt
        self.lr = args.learning_rate
        self.epochs = args.epochs
        self.log_intermediate = args.log_intermediate
        self.log_symplectic_loss = args.log_symplectic_loss
        self.log_spatial_loss = args.log_spatial_loss
        self.model = model
        self.surrogate_model = surrogate_model
        self.output_dir = args.output_dir

        if args.integrator == 'implicit_midpoint':
            self.num_integrator = ImplicitMidpointIntegrator(self.dt)
        elif args.integrator == 'stoermer_verlet_q':
            self.num_integrator = SeparableStormerVerletIntegrator(self.dt, use_staggered='q')
        elif args.integrator == 'stoermer_verlet_p':
            self.num_integrator = SeparableStormerVerletIntegrator(self.dt, use_staggered='p')

        self._sampling_params = sampling_params

        self._configurations = []
        self._init_writer(args)

        # save args on disk
        save_json((self.output_dir, 'args.json'), vars(args))
        save_json((self.output_dir, 'stats.json'), {
            'learnable_parameters': sum(p.numel() for p in self.surrogate_model.parameters() if p.requires_grad)
        })

    def _init_writer(self, args):
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tensorboard'), comment='lowdim')
        self.writer.add_text('hyperparameters', str(args))

    def _get_training_data(self):
        params = self._sampling_params

        # do not change original dictionary
        mu = params.mu.copy()

        # generate random phase points in [-qmin,qmax]x[-pmin,pmax]
        rg = np.random.default_rng(seed=0)
        q = rg.uniform(params.qmin, params.qmax, size=(params.n,1))
        p = rg.uniform(params.pmin, params.pmax, size=(params.n,1))
        X_train = np.hstack((q,p))

        def time_step(x_train):
            # TODO implement generic way to set initial values
            mu['q0'] = x_train[0]
            mu['p0'] = x_train[1]

            # compute single time step h
            td_x, td_Ham = model.solve(0, self.dt+self.dt, 
                self.num_integrator, mu) # open time interval, therefore `dt+dt`
            t, x = td_x[1]
            return x.to_numpy()

        Y_train = np.apply_along_axis(time_step, 1, X_train)

        # convert numpy array to `torch.tensor`
        X_train = torch.tensor(X_train, requires_grad=True, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.float32)

        return (X_train, Y_train)

    def _run_configurations(self, epoch):
        for configuration in self._configurations:
            configuration.run(epoch)

    def _log_loss(self, loss, x, y1, y, epoch):
        self.writer.add_scalar("Loss/train", loss, epoch)

        if self.log_symplectic_loss:
            # calculate and log symplectic loss
            symplectic_loss = symplectic_mse_loss(x, y1)
            self.writer.add_scalar('Loss/symplectic', symplectic_loss, epoch)

        if self.log_spatial_loss:
            # calculate and log spatial loss
            diff = torch.norm(y-y1, p=2, dim=1)
            plt_loss = plot_spatial_error(x.detach().numpy(), diff.detach().numpy())
            self.writer.add_figure('Loss/Spatial', plt_loss, epoch)

    def add_configuration(self, name, mu, t_start, t_end):
        self._configurations.append(Configuration(self, name, mu, t_start, t_end))

    def run(self):
        x,y = self._get_training_data()

        # save training data on disk
        np.save(os.path.join(self.output_dir, 'training_data_x.npy'), x.detach())
        np.save(os.path.join(self.output_dir, 'training_data_y.npy'), y.detach())

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=self.lr, amsgrad=True)

        losses = []
        self.surrogate_model.train()
        for epoch in range(self.epochs):
            if (epoch % 100) == 0:
                print('training step: %d/%d' % (epoch, self.epochs))
    
            y1 = self.surrogate_model(x)
            loss = criterion(y1, y)        
            optimizer.zero_grad()

            # retain_graph=True to calculate symplectic loss afterwards
            loss.backward(retain_graph=self.log_symplectic_loss)
            optimizer.step()

            with torch.no_grad():
                self.surrogate_model.train(mode=False)

                self._log_loss(loss, x, y1, y, epoch)
                losses.append(float(loss))
                if self.log_intermediate and (epoch % 500) == 0:
                    self._run_configurations(epoch)

                self.surrogate_model.train()

        self.surrogate_model.train(mode=False)
        self._run_configurations(self.epochs)
        self.writer.add_graph(self.surrogate_model, x)
        self.writer.flush()

        # save losses and trained model on disk
        np.save(os.path.join(self.output_dir, 'losses.npy'), losses)
        torch.save(self.surrogate_model.state_dict(), 
            os.path.join(self.output_dir, 'surrogate_model.state_dict'))

def get_surrogate_model(architecture, dim):
    if architecture == 'la-sympnet':
        return torch.nn.Sequential(
            LinearSymplectic(4, dim, bias=True),
            LowerNonlinearSymplectic(dim, bias=False, activation_fn=activation_fn),
            LinearSymplectic(4, dim, bias=True),
            UpperNonlinearSymplectic(dim, bias=False, activation_fn=activation_fn),
            LinearSymplectic(4, dim, bias=True),
            LowerNonlinearSymplectic(dim, bias=False, activation_fn=activation_fn),
            LinearSymplectic(4, dim, bias=True),
            UpperNonlinearSymplectic(dim, bias=False, activation_fn=activation_fn),
            LinearSymplectic(4, dim, bias=True),
            LowerNonlinearSymplectic(dim, bias=False, activation_fn=activation_fn),
            LinearSymplectic(4, dim, bias=True)
        )
    elif architecture == 'normalized-la-sympnet':
        return torch.nn.Sequential(
            LinearSymplectic(4, dim, bias=True),
            NormalizedLowerNonlinearSymplectic(dim, bias=False, activation_fn=activation_fn),
            LinearSymplectic(4, dim, bias=True),
            NormalizedUpperNonlinearSymplectic(dim, bias=False, activation_fn=activation_fn),
            LinearSymplectic(4, dim, bias=True),
            NormalizedLowerNonlinearSymplectic(dim, bias=False, activation_fn=activation_fn),
            LinearSymplectic(4, dim, bias=True),
            NormalizedUpperNonlinearSymplectic(dim, bias=False, activation_fn=activation_fn),
            LinearSymplectic(4, dim, bias=True),
            NormalizedLowerNonlinearSymplectic(dim, bias=False, activation_fn=activation_fn),
            LinearSymplectic(4, dim, bias=True)
        )
    elif architecture == 'g-sympnet':
        return torch.nn.Sequential(
            LowerGradientModule(dim, n=30, bias=False, activation_fn=activation_fn),
            UpperGradientModule(dim, n=30, bias=False, activation_fn=activation_fn),
            LowerGradientModule(dim, n=30, bias=False, activation_fn=activation_fn),
            UpperGradientModule(dim, n=30, bias=False, activation_fn=activation_fn)
        )
    elif architecture == 'normalized-g-sympnet':
        return torch.nn.Sequential(
            NormalizedLowerGradientModule(dim, n=30, bias=False, activation_fn=activation_fn),
            NormalizedUpperGradientModule(dim, n=30, bias=False, activation_fn=activation_fn),
            NormalizedLowerGradientModule(dim, n=30, bias=False, activation_fn=activation_fn),
            NormalizedUpperGradientModule(dim, n=30, bias=False, activation_fn=activation_fn)
        )
    else:
        raise ValueError('Invalid architecture {}'.format(architecture))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--model', choices=['harmonic_oscillator', 'simple_pendulum'])
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--dt', default=0.1, type=float)
    parser.add_argument('--training-size', '-n', default=40, type=int)
    parser.add_argument('--learning-rate', '-lr', default=1e-2, type=float)
    parser.add_argument(
        '--integrator',
        help='Choose integrator.',
        choices=['implicit_midpoint', 'stoermer_verlet_q', 'stoermer_verlet_p'],
        default='stoermer_verlet_q'
    )
    parser.add_argument('--architecture', choices=['la-sympnet', 'normalized-la-sympnet', 'g-sympnet', 'normalized-g-sympnet'], default='la-sympnet')
    parser.add_argument('--activation', choices=['sigmoid', 'sin', 'relu', 'elu', 'snake'], default='sigmoid')
    parser.add_argument('--qmin', default=-np.pi/2, type=float)
    parser.add_argument('--qmax', default=np.pi/2, type=float)
    parser.add_argument('--pmin', default=-np.sqrt(2), type=float)
    parser.add_argument('--pmax', default=np.sqrt(2), type=float)
    parser.add_argument('--log-intermediate', default=False, nargs='?', const=True, type=bool)
    parser.add_argument('--log-symplectic-loss', default=False, nargs='?', const=True, type=bool)
    parser.add_argument('--log-spatial-loss', default=False, nargs='?', const=True, type=bool)
    args = parser.parse_args()

    activation_functions = {
        'sigmoid': torch.sigmoid,
        'sin': torch.sin,
        'relu': torch.relu,
        'elu': torch.nn.ELU(),
        'snake': lambda x: x+torch.sin(x)**2 # https://arxiv.org/pdf/2006.08195v1.pdf
    }
    activation_fn = activation_functions[args.activation]
    dim = 2
    surrogate_model = get_surrogate_model(args.architecture, dim)

    if args.model == 'simple_pendulum':
        model = SimplePendulum()

        mu = {'m': 1., 'g': 1., 'l': 1., 'q0': np.pi/2, 'p0': 0.}
        sampling_params = SamplingParameters(mu,
            qmin=args.qmin, qmax=args.qmax,
            pmin=args.pmin, pmax=args.pmax)
        sampling_params.n = args.training_size

        expm = Experiment(args, model, surrogate_model, sampling_params)

        expm.add_configuration('swinging_case',
            {'m': 1., 'g': 1., 'l': 1., 'q0': np.pi/2, 'p0': 0.}, 
            t_start = 0, t_end = 100)

        expm.add_configuration('rotating_case',
            {'m': 1., 'g': 1., 'l': 1., 'q0': np.pi, 'p0': 1.}, 
            t_start = 0, t_end = 10)

        expm.run()
    
    elif args.model == 'harmonic_oscillator':
        model = HarmonicOscillator()

        mu = {'m': 1., 'k': 1., 'f': 0., 'q0': 1., 'p0': 0.}
        sampling_params = SamplingParameters(mu,
            qmin=args.qmin, qmax=args.qmax,
            pmin=args.pmin, pmax=args.pmax)
        sampling_params.n = args.training_size

        expm = Experiment(args, model, surrogate_model, sampling_params)

        expm.add_configuration('harmonic_oscillator',
            {'m': 1., 'k': 1., 'f': 0., 'q0': 1., 'p0': 0.}, 
            t_start = 0, t_end = 100)

        expm.run()

    else:
        raise ValueError('Invalid model {}'.format(args.model))