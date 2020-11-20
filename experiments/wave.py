import argparse
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.vectors import NumpyPhaseSpace
from models.wave import (OscillatingModeLinearWaveProblem, FixedEndsLinearWaveProblem,
    SineGordonProblem)
from models.integrators.implicit_midpoint import ImplicitMidpointIntegrator
from models.integrators.stormer_verlet import SeparableStormerVerletIntegrator

from nn.models import integrate
from nn.linearsymplectic import *
from nn.nonlinearsymplectic import *
from nn.training_data import scale_training_data

from utils.plot2d import plot_hamiltonian
from utils.plot_wave import *
from utils.data import td_x_to_hamiltonian, save_json, td_to_numpy

class Dirichlet1dNumpyPhaseSpace(NumpyPhaseSpace):
    def __init__(self, dim, q_left=0, q_right=0, p_left=0, p_right=0):
        super(Dirichlet1dNumpyPhaseSpace, self).__init__(dim+4)
        self.q_left = q_left
        self.q_right = q_right
        self.p_left = p_left
        self.p_right = p_right

    def new_vector(self, vec_q, vec_p):
        # insert Dirichlet values at boundaries
        vec_q = np.insert(vec_q, 0, self.q_left)
        vec_q = np.insert(vec_q, len(vec_q), self.q_right)

        vec_p = np.insert(vec_p, 0, self.p_left)
        vec_p = np.insert(vec_p, len(vec_p), self.p_right)

        return super().new_vector(vec_q, vec_p)

class ActivationModule(torch.nn.Module):
    def __init__(self, activation_fn):
        super(ActivationModule, self).__init__()
        self.activation_fn = activation_fn

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(input)

class WaveExperiment:
    def __init__(self, args):
        self.epochs = args.epochs
        self.n_x = args.nx
        self.print_params = args.print_params
        self.dt = args.dt
        self.T = args.time_total
        self.T_training = args.time_training
        self.post_plot_transport = args.post_plot_transport
        self.log_intermediate = args.log_intermediate
        self.output_dir = args.output_dir
        self.dim = 2*self.n_x-4 # -4 because Dirichlet boundary values removed
        
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tensorboard'),comment='wave')

        self.l = args.domain_length
        self._init_model(args.model)

        if args.integrator == 'implicit_midpoint':
            self.num_integrator = ImplicitMidpointIntegrator(self.dt)
        elif args.integrator == 'stoermer_verlet_q':
            self.num_integrator = SeparableStormerVerletIntegrator(self.dt, use_staggered='q')
        elif args.integrator == 'stoermer_verlet_p':
            self.num_integrator = SeparableStormerVerletIntegrator(self.dt, use_staggered='p')
        
        if args.init_stormer_verlet:
            self._init_with_stormer_verlet(args.model)
        else:
            self._init_surrogate_model(args.architecture, args.activation)

        # save args on disk
        save_json((self.output_dir, 'args.json'), vars(args))
        save_json((self.output_dir, 'stats.json'), {
            'learnable_parameters': sum(p.numel() for p in self.surrogate_model.parameters() if p.requires_grad)
        })

    def _init_surrogate_model(self, architecture, activation):
        activation_functions = {
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'elu': torch.nn.ELU(),
        }
        activation_fn = activation_functions[args.activation]

        if architecture == 'linear_canonical' or architecture  == 'linear_fd':
            kernel_basis = (FDSymmetricKernelBasis(kernel_size = 3) if architecture == 'linear_fd'
                else CanonicalSymmetricKernelBasis(kernel_size=3))
            
            self.surrogate_model = torch.nn.Sequential(
                UpperSymplecticConv1d(self.dim, kernel_basis=kernel_basis),
                LowerSymplecticConv1d(self.dim, kernel_basis=kernel_basis),
                UpperSymplecticConv1d(self.dim, kernel_basis=kernel_basis),
                LowerSymplecticConv1d(self.dim, kernel_basis=kernel_basis),
                UpperSymplecticConv1d(self.dim, kernel_basis=kernel_basis),
                LowerSymplecticConv1d(self.dim, kernel_basis=kernel_basis),
                UpperSymplecticConv1d(self.dim, kernel_basis=kernel_basis),
                LowerSymplecticConv1d(self.dim, kernel_basis=kernel_basis)
            )
        elif architecture == 'nonlinear':
            kernel_basis = FDSymmetricKernelBasis(kernel_size = 3)
            self.surrogate_model = torch.nn.Sequential(
                UpperSymplecticConv1d(self.dim, padding_mode='replicate', kernel_basis=kernel_basis),
                UpperNonlinearSymplectic(self.dim, activation_fn=torch.sin, scalar_weight=True),

                LowerSymplecticConv1d(self.dim, padding_mode='replicate', kernel_basis=kernel_basis),
                LowerNonlinearSymplectic(self.dim, activation_fn=torch.sin, scalar_weight=True),

                UpperSymplecticConv1d(self.dim, padding_mode='replicate', kernel_basis=kernel_basis),
                UpperNonlinearSymplectic(self.dim, activation_fn=torch.sin, scalar_weight=True),

                LowerSymplecticConv1d(self.dim, padding_mode='replicate', kernel_basis=kernel_basis),
                LowerNonlinearSymplectic(self.dim, activation_fn=torch.sin, scalar_weight=True)
            )
        elif architecture == 'gradient':
            kernel_basis = FDSymmetricKernelBasis(kernel_size = 3)
            gradient_args = {'dim': self.dim, 'bias': False, 'n': 100, 'activation_fn': activation_fn}
            self.surrogate_model = torch.nn.Sequential(
                UpperSymplecticConv1d(self.dim, padding_mode='replicate', kernel_basis=kernel_basis),
                UpperConv1dGradientModule(**gradient_args),

                LowerSymplecticConv1d(self.dim, padding_mode='replicate', kernel_basis=kernel_basis),
                LowerConv1dGradientModule(**gradient_args),

                UpperSymplecticConv1d(self.dim, padding_mode='replicate', kernel_basis=kernel_basis),
                UpperConv1dGradientModule(**gradient_args),

                LowerSymplecticConv1d(self.dim, padding_mode='replicate', kernel_basis=kernel_basis),
                LowerConv1dGradientModule(**gradient_args)
            )
        elif architecture == 'n1-gradient' or architecture == 'n2-gradient':
            kernel_basis = FDSymmetricKernelBasis(kernel_size = 3)
            ignore_factor = architecture == 'n1-gradient'
            gradient_args = {'dim': self.dim, 'bias': False, 'n': 100, 'activation_fn': activation_fn,
                'ignore_factor': ignore_factor}
            
            self.surrogate_model = torch.nn.Sequential(
                UpperSymplecticConv1d(self.dim, padding_mode='replicate', kernel_basis=kernel_basis),
                NormalizedUpperConv1dGradientModule(**gradient_args),

                LowerSymplecticConv1d(self.dim, padding_mode='replicate', kernel_basis=kernel_basis),
                NormalizedLowerConv1dGradientModule(**gradient_args),

                UpperSymplecticConv1d(self.dim, padding_mode='replicate', kernel_basis=kernel_basis),
                NormalizedUpperConv1dGradientModule(**gradient_args),

                LowerSymplecticConv1d(self.dim, padding_mode='replicate', kernel_basis=kernel_basis),
                NormalizedLowerConv1dGradientModule(**gradient_args)
            )
        elif architecture == 'cnn':
            conv1d_args = {'in_channels': 1, 'out_channels': 1, 'kernel_size': 3, 
                'padding': 1, 'padding_mode': 'replicate', 'bias': False}

            upscale_args = {'in_channels': 1, 'out_channels': 1, 
                'kernel_size': 100, 'stride': 100, 'bias': False}

            self.surrogate_model = torch.nn.Sequential(
                torch.nn.Conv1d(**conv1d_args),
                torch.nn.ConvTranspose1d(**upscale_args),
                ActivationModule(activation_fn=activation_fn),
                torch.nn.Conv1d(**upscale_args),

                torch.nn.Conv1d(**conv1d_args),
                torch.nn.ConvTranspose1d(**upscale_args),
                ActivationModule(activation_fn=activation_fn),
                torch.nn.Conv1d(**upscale_args),

                torch.nn.Conv1d(**conv1d_args),
                torch.nn.ConvTranspose1d(**upscale_args),
                ActivationModule(activation_fn=activation_fn),
                torch.nn.Conv1d(**upscale_args),

                torch.nn.Conv1d(**conv1d_args),
                torch.nn.ConvTranspose1d(**upscale_args),
                ActivationModule(activation_fn=activation_fn),
                torch.nn.Conv1d(**upscale_args)
            )


    def _init_with_stormer_verlet(self, model_name):
        if model_name == 'standing_wave' or model_name == 'transport':
            kernel_basis = FDSymmetricKernelBasis(kernel_size = 3)
            self.surrogate_model = torch.nn.Sequential(
                UpperSymplecticConv1d(self.dim, kernel_basis=kernel_basis),
                LowerSymplecticConv1d(self.dim, kernel_basis=kernel_basis),
                UpperSymplecticConv1d(self.dim, kernel_basis=kernel_basis)
            )

            self.surrogate_model[0].a.data = self.surrogate_model[2].a.data = torch.tensor([
                [self.dt/2],
                [0.]
            ])

            dx = self.l/self.n_x
            self.surrogate_model[1].a.data = torch.tensor([
                [0.],
                [self.dt*(self.mu['c']/dx)**2]
            ])
        elif model_name == 'sine_gordon':
            kernel_basis = FDSymmetricKernelBasis(kernel_size = 3)

            # non-zero Dirichlet value
            initial_value = self.model.initial_value(self.mu)
            q_left = initial_value.vec_q[0]
            q_right = initial_value.vec_q[-1]
            padding = (q_left, q_right)

            self.surrogate_model = torch.nn.Sequential(
                UpperSymplecticConv1d(self.dim, kernel_basis=kernel_basis),
                LowerSymplecticConv1d(self.dim, padding_values=padding, kernel_basis=kernel_basis),
                LowerNonlinearSymplectic(self.dim, activation_fn=torch.sin, scalar_weight=True),
                UpperSymplecticConv1d(self.dim, kernel_basis=kernel_basis)
            )

            self.surrogate_model[0].a.data = self.surrogate_model[3].a.data = torch.tensor([
                [self.dt/2],
                [0.]
            ])

            dx = self.l/self.n_x
            self.surrogate_model[1].a.data = torch.tensor([
                [0.], 
                [self.dt*(self.mu['c']/dx)**2]
            ])

            self.surrogate_model[2].a.data = torch.tensor([-self.dt])

    def _init_model(self, model_name):
        if model_name == 'standing_wave':
            self.mu = {'c': .5}
            self.model = OscillatingModeLinearWaveProblem(self.l, self.n_x)
            self.surrogate_phase_space = Dirichlet1dNumpyPhaseSpace(self.dim)
        elif model_name == 'transport':
            self.mu = {'c': .1, 'q0_supp': self.l/4}
            self.model = FixedEndsLinearWaveProblem(self.l, self.n_x)
            self.surrogate_phase_space = Dirichlet1dNumpyPhaseSpace(self.dim)
        elif model_name == 'sine_gordon':
            self.mu = {'c': 1, 'v': .2}
            self.model = SineGordonProblem(self.l, self.n_x)
            initial_value = self.model.initial_value(self.mu)
            q_right = initial_value.vec_q[-1]
            self.surrogate_phase_space = Dirichlet1dNumpyPhaseSpace(self.dim, q_right = q_right)

    def _compute_solution(self): 
        # compute solution for all t in [0, T]
        self.td_x, self.td_Ham = self.model.solve(0, self.T, self.num_integrator, self.mu)

        # save exact solution on disk
        np.save(os.path.join(self.output_dir, 'exact_td_x.npy'), 
            td_to_numpy(self.td_x))
        np.save(os.path.join(self.output_dir, 'exact_td_Ham.npy'), 
            td_to_numpy(self.td_Ham))

    def _get_training_data(self):
        assert hasattr(self, 'td_x'), 'Call _compute_solution() before calling _get_training_data()'
        data = self.td_x.all_data_to_numpy()

        # delete Dirichlet boundary values from training data
        data = np.delete(data, [0, self.n_x-1, self.n_x, 2*self.n_x-1], axis=1)

        n_training = int(self.T_training / self.dt)
        x = data[0:n_training,:]
        y = data[1:n_training+1,:] # shift by 1
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        x_test = data[n_training+1:-1]
        y_test = data[n_training+2:] # shift by 1
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        return x,y,x_test,y_test

    def _plot(self, epoch, prefix='', other_model=None, other_mu = None):
        this_model = self.model if other_model is None else other_model
        this_mu = self.mu if other_mu is None else other_mu

        this_td_Ham = self.td_Ham
        this_td_x = self.td_x
        if other_model is not None or other_mu is not None:
            this_td_x, this_td_Ham = this_model.solve(0, self.T, self.num_integrator, this_mu)

        x0 = this_model.initial_value(this_mu)
        td_x_surrogate = integrate(self.surrogate_model, x0.vec_q[1:-1], x0.vec_p[1:-1], 
            0, self.T, self.dt, custom_phase_space=self.surrogate_phase_space)
        td_Ham_surrogate = td_x_to_hamiltonian(td_x_surrogate, self.model, this_mu)

        # save data on disk
        np.save(os.path.join(self.output_dir, 'epoch{}_td_x.npy'.format(epoch)), 
            td_to_numpy(td_x_surrogate))
        np.save(os.path.join(self.output_dir, 'epoch{}_td_Ham.npy'.format(epoch)), 
            td_to_numpy(td_Ham_surrogate))

        # Plot Hamiltonian
        ham_plt = plot_hamiltonian(this_td_Ham, td_Ham_surrogate)
        self.writer.add_figure(prefix + '/Hamiltonian', ham_plt, epoch) 

        # Plot q and p for left, middle and right knot
        q_left = plot_q_over_time(this_td_x, td_x_surrogate, 0)
        self.writer.add_figure(prefix + 'left/q', q_left, epoch)
        p_left = plot_p_over_time(this_td_x, td_x_surrogate, 0)
        self.writer.add_figure(prefix + 'left/p', p_left, epoch)
        
        q_mid = plot_q_over_time(this_td_x, td_x_surrogate, int(self.n_x/2))
        self.writer.add_figure(prefix + 'middle/q', q_mid, epoch)
        p_mid = plot_p_over_time(this_td_x, td_x_surrogate, int(self.n_x/2))
        self.writer.add_figure(prefix + 'middle/p', p_mid, epoch)

        q_right = plot_q_over_time(this_td_x, td_x_surrogate, self.n_x-1)
        self.writer.add_figure(prefix + 'right/q', q_right, epoch)
        p_right = plot_p_over_time(this_td_x, td_x_surrogate, self.n_x-1)
        self.writer.add_figure(prefix + 'right/p', p_right, epoch)

        # Plot q and p for t=1, t=5 and t=9
        for t in [0, self.dt, 1, 5, 9]:
            q_t = plot_q_over_domain(self, this_td_x, td_x_surrogate, t)
            self.writer.add_figure(prefix + 't{}/q'.format(t), q_t, epoch)
            p_t = plot_p_over_domain(self, this_td_x, td_x_surrogate, t)
            self.writer.add_figure(prefix + 't{}/p'.format(t), p_t, epoch)
        
    def run(self):
        self._compute_solution()
        x, y, x_test, y_test = self._get_training_data()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(self.surrogate_model.parameters(), lr=1e-4, eps=1e-6, amsgrad=True)

        losses = []
        test_losses = []
        self.surrogate_model.train()
        for epoch in range(self.epochs):  
            y1 = self.surrogate_model(x)
            loss = criterion(y1, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss))
            self.writer.add_scalar("Loss/train", loss, epoch)

            self.surrogate_model.train(mode=False)  

            with torch.no_grad():
                y1_test = self.surrogate_model(x_test)
                test_loss = criterion(y1_test, y_test)
                test_losses.append(float(test_loss))
                self.writer.add_scalar('Loss/test', test_loss, epoch)

                if (epoch % 100) == 0:
                    print('training step: {}/{}, training loss: {}, test loss: {}'
                        .format(epoch, self.epochs, float(loss), float(test_loss)))
                      
                if self.log_intermediate and (epoch % 500 == 0): self._plot(epoch)
            
            self.surrogate_model.train()

        self.surrogate_model.train(mode=False)
        self._plot(self.epochs)
        self.writer.add_graph(self.surrogate_model, x)

        if self.post_plot_transport:
            self._plot(None, 'transport/', 
                FixedEndsLinearWaveProblem(self.l, self.n_x),
                {'c': .5, 'q0_supp': self.l/4})

        self.writer.flush()

        if self.print_params:
            for name, param in self.surrogate_model.named_parameters():
                if param.requires_grad:
                    print(name, param.data)

        # save losses and trained model on disk
        np.save(os.path.join(self.output_dir, 'losses.npy'), losses)
        np.save(os.path.join(self.output_dir, 'test_losses.npy'), test_losses)
        torch.save(self.surrogate_model.state_dict(), 
            os.path.join(self.output_dir, 'surrogate_model.state_dict'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--architecture', choices=
        [
            'linear_canonical', 
            'linear_fd', 
            'nonlinear', 
            'gradient',
            'n1-gradient',
            'n2-gradient',
            'cnn'
        ], 
        default='linear',
        help='The architecture to use for the neural network.')
    parser.add_argument('--activation', 
        choices=['sigmoid', 'tanh', 'elu'], 
        default='sigmoid')
    parser.add_argument('--model', choices=['standing_wave', 'transport', 'sine_gordon'], default='standing_wave')
    parser.add_argument('--dt', default=.1, type=float)
    parser.add_argument('--time-total', default=10.0, type=float)
    parser.add_argument('--time-training', default=0.1, type=float)
    parser.add_argument(
        '--integrator',
        help='Choose integrator.',
        choices=['implicit_midpoint', 'stoermer_verlet_q', 'stoermer_verlet_p'],
        default='stoermer_verlet_q'
    )
    parser.add_argument('--nx', default=100, type=int)
    parser.add_argument('--domain-length', '-l', default=1, type=float)
    parser.add_argument('--print-params', default=False, nargs='?', const=True, type=bool)
    parser.add_argument('--post-plot-transport', default=False, nargs='?', const=True, type=bool, 
        help='Plot transport problem after training.')
    parser.add_argument('--init-stormer-verlet', default=False, nargs='?', const=True, type=bool, 
        help='Initialize weights with Stormer-Verlet integration scheme.')
    parser.add_argument('--log-intermediate', default=False, nargs='?', const=True, type=bool)
    args = parser.parse_args()

    expm = WaveExperiment(args)
    expm.run()