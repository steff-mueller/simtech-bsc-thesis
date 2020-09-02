import argparse

import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

from models.lowdim import SimplePendulum, HarmonicOscillator

from nn.training_data import generate_training_data
from nn.models import SympNet
from nn.symplecticloss import symplectic_mse_loss

from utils.plot2d import *

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
        self._td_x, self._td_Ham = self.experiment.model.solve(self.t_start, self.t_end, self.experiment.dt, self.mu)

    def run(self, epoch):
        td_x_surrogate = self.experiment.surrogate_model.integrate(self.mu['q0'], self.mu['p0'], 
            self.t_start, self.t_end, self.experiment.dt)

        # plot results
        plt = plot_Ham_sys(self._td_x, td_x_surrogate)
        self.experiment.writer.add_figure(self.name + '/PhaseSpace', plt, epoch)

        qplt = plot_q(self._td_x, td_x_surrogate)
        self.experiment.writer.add_figure(self.name + '/q', qplt, epoch)

        pplt = plot_p(self._td_x, td_x_surrogate)
        self.experiment.writer.add_figure(self.name + '/p', pplt, epoch)

        ham_plt = plot_hamiltonian(self._td_Ham, td_x_surrogate, self.experiment.model, self.mu)
        self.experiment.writer.add_figure(self.name + '/Hamiltonian', ham_plt, epoch) 

"""
    Plot equilibriums for simple pendulum
"""
class EquilibriumConfiguration:
    def __init__(self, experiment, name):
        self.experiment = experiment
        self.name = name

    def run(self, epoch):
        n = 100
        idx = torch.arange(-n, n+1)
        q = math.pi*idx
        p = torch.zeros_like(q)
        x_equilibrium = torch.stack([q,p],dim=1)
        y_equilibrium = self.experiment.surrogate_model(x_equilibrium).detach().numpy()
        diff = x_equilibrium - y_equilibrium
        diff_q = diff[:,0]
        diff_p = diff[:,1]

        fig = plt.figure()
        ax = plt.axes()
        ax.plot(idx, diff_q)
        ax.plot(idx, diff_p)
        ax.legend([ 
            'diff q',
            'diff p'
        ])
        self.experiment.writer.add_figure(self.name, fig, epoch)

class Experiment:

    def __init__(self, args, model, surrogate_model):
        self.dt = args.dt
        self.epochs = args.epochs
        self.model = model
        self.surrogate_model = surrogate_model

        mu = {'m': 1., 'g': 1., 'l': 1., 'q0': np.pi/2, 'p0': 0.}
        self._sampling_params = SamplingParameters(mu,
            qmin=args.qmin, qmax=args.qmax,
            pmin=args.pmin, pmax=args.pmax)
        self._sampling_params.n = args.training_size

        self._configurations = []
        self._init_writer(args)

    def _init_writer(self, args):
        self.writer = SummaryWriter(comment='lowdim')
        self.writer.add_text('hyperparameters', str({
            'model': self.model.__class__.__name__,
            'surrogate_model': self.surrogate_model.__class__.__name__
        }) + ' ' + str(args))

    def _get_training_data(self):
        return generate_training_data(self._sampling_params.n, 
            self.model, 
            self._sampling_params.mu, 
            self.dt, 
            qmin=self._sampling_params.qmin, 
            qmax=self._sampling_params.qmax,
            pmin=self._sampling_params.pmin, 
            pmax=self._sampling_params.pmax)

    def _run_configurations(self, epoch):
        for configuration in self._configurations:
            configuration.run(epoch)

    def _log_loss(self, loss, x, y1, y, epoch):
        self.writer.add_scalar("Loss/train", loss, epoch)

        # calculate and log symplectic loss
        symplectic_loss = symplectic_mse_loss(x, y1)
        self.writer.add_scalar('Loss/symplectic', symplectic_loss, epoch)

        # calculate and log spatial loss
        diff = torch.norm(y-y1, p=2, dim=1)
        plt_loss = plot_spatial_error(x.detach().numpy(), diff.detach().numpy())
        self.writer.add_figure('Loss/Spatial', plt_loss, epoch)

    def add_configuration(self, name, mu, t_start, t_end):
        self._configurations.append(Configuration(self, name, mu, t_start, t_end))

    def run(self):
        x,y = self._get_training_data()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=1e-2)

        for epoch in range(self.epochs):
            print('training step: %d/%d' % (epoch, self.epochs))
    
            y1 = self.surrogate_model(x)
            loss = criterion(y1, y)        
            optimizer.zero_grad()

            # retain_graph=True to calculate symplectic loss afterwards
            loss.backward(retain_graph=True)
            self._log_loss(loss, x, y1, y, epoch)

            optimizer.step()

            if (epoch % 200) == 0:
                self._run_configurations(epoch)

        self._run_configurations(self.epochs)
        self.writer.add_graph(self.surrogate_model, x)
        self.writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--dt', default=0.1, type=float)
    parser.add_argument('--training-size', '-n', default=40, type=int)
    parser.add_argument('--activation', choices=['sigmoid', 'sin', 'relu'], default='sigmoid')
    parser.add_argument('--qmin', default=-np.pi/2, type=float)
    parser.add_argument('--qmax', default=np.pi/2, type=float)
    parser.add_argument('--pmin', default=-np.sqrt(2), type=float)
    parser.add_argument('--pmax', default=np.sqrt(2), type=float)
    args = parser.parse_args()

    model = SimplePendulum()

    activation_functions = {
        'sigmoid': torch.sigmoid,
        'sin': torch.sin,
        'relu': torch.relu
    }
    activation_fn = activation_functions[args.activation]
    surrogate_model = SympNet(layers = 5, sub_layers = 4, dim = 2, activation_fn=activation_fn)

    expm = Experiment(args, model, surrogate_model)

    expm.add_configuration('swinging_case',
        {'m': 1., 'g': 1., 'l': 1., 'q0': np.pi/2, 'p0': 0.}, 
        t_start = 0, t_end = 100)

    expm.add_configuration('rotating_case',
        {'m': 1., 'g': 1., 'l': 1., 'q0': np.pi, 'p0': 1.}, 
        t_start = 0, t_end = 100)

    expm.add_configuration('stationary_stable_case',
        {'m': 1., 'g': 1., 'l': 1., 'q0': 0, 'p0': 0.},
        t_start = 0, t_end = 10)
    expm.add_configuration('stationary_unstable_case',
        {'m': 1., 'g': 1., 'l': 1., 'q0': np.pi, 'p0': 0.},
        t_start = 0, t_end = 10)

    expm._configurations.append(EquilibriumConfiguration(expm, 'equilibriums'))

    expm.run()

    # model = HarmonicOscillator()
    # dt = 0.1
    # q0 = 1.
    # p0 = 0.
    # mu = {'m': 1., 'k': 1., 'f': 0., 'q0': q0, 'p0': p0}
    # x, y = generate_training_data(10000, model, mu, dt, 
    #   qmin = -2, qmax = 2,
    #   pmin = -2, pmax = 2)

    #surrogate_model = HarmonicSympNet(layers = 8, sub_layers = 5, dim = 1, dt = 0.1)