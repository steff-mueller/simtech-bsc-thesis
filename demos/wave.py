import argparse

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

class Dirichlet1dNumpyPhaseSpace(NumpyPhaseSpace):
    def __init__(self, dim):
        super(Dirichlet1dNumpyPhaseSpace, self).__init__(dim+4)

    def new_vector(self, vec_q, vec_p):
        # insert zero Dirichlet values at boundaries
        vec_q = np.insert(vec_q, [0, len(vec_q)], 0)
        vec_p = np.insert(vec_p, [0, len(vec_p)], 0)

        return super().new_vector(vec_q, vec_p)

class WaveExperiment:
    def __init__(self, args):
        self.epochs = args.epochs
        self.n_x = args.nx
        self.print_params = args.print_params
        self.dt = args.dt
        self.post_plot_transport = args.post_plot_transport
        
        self.writer = SummaryWriter(comment='wave')
        # TODO add parameters logging

        self.l = args.domain_length
        self._init_model(args.model)

        if args.integrator == 'implicit_midpoint':
            self.num_integrator = ImplicitMidpointIntegrator(self.dt)
        elif args.integrator == 'stoermer_verlet_q':
            self.num_integrator = SeparableStormerVerletIntegrator(self.dt, use_staggered='q')
        elif args.integrator == 'stoermer_verlet_p':
            self.num_integrator = SeparableStormerVerletIntegrator(self.dt, use_staggered='p')

        self.dim = 2*self.n_x-4 # -4 because Dirichlet boundary values removed
        kernel_basis = FDSymmetricKernelBasis(kernel_size = 3)
        self.surrogate_model = torch.nn.Sequential(
            UpperSymplecticConv1d(self.dim, bias=False, kernel_basis=kernel_basis),
            LowerSymplecticConv1d(self.dim, bias=False, kernel_basis=kernel_basis),
            UpperSymplecticConv1d(self.dim, bias=False, kernel_basis=kernel_basis),
            LowerSymplecticConv1d(self.dim, bias=False, kernel_basis=kernel_basis),
            UpperSymplecticConv1d(self.dim, bias=False, kernel_basis=kernel_basis),
            LowerSymplecticConv1d(self.dim, bias=False, kernel_basis=kernel_basis),
            UpperSymplecticConv1d(self.dim, bias=False, kernel_basis=kernel_basis),
            LowerSymplecticConv1d(self.dim, bias=False, kernel_basis=kernel_basis)
        )

        if args.init_stormer_verlet:
            self._init_with_stormer_verlet()
    
    def _init_with_stormer_verlet(self):
        kernel_basis = FDSymmetricKernelBasis(kernel_size = 3)
        self.surrogate_model = torch.nn.Sequential(
            UpperSymplecticConv1d(self.dim, bias=False, kernel_basis=kernel_basis),
            LowerSymplecticConv1d(self.dim, bias=False, kernel_basis=kernel_basis),
            UpperSymplecticConv1d(self.dim, bias=False, kernel_basis=kernel_basis)
        )

        # initialize weights with Störmer-Verlet method
        self.surrogate_model[0].a.data = self.surrogate_model[2].a.data = torch.tensor([
            [self.dt/2],
            [0.]
        ])

        dx = self.l/self.n_x
        self.surrogate_model[1].a.data = torch.tensor([
            [0.], 
            [self.dt*(self.mu['c']/dx)**2]
        ])

    def _init_model(self, model_name):
        if model_name == 'standing_wave':
            self.mu = {'c': .5}
            self.model = OscillatingModeLinearWaveProblem(self.l, self.n_x)
        elif model_name == 'transport':
            self.mu = {'c': .1, 'q0_supp': self.l/4}
            self.model = FixedEndsLinearWaveProblem(self.l, self.n_x)
        elif model_name == 'sine_gordon':
            self.mu = {'c': 1, 'v': .2}
            self.model = SineGordonProblem(self.l, self.n_x)

    def _compute_solution(self): 
        # compute solution for all t in [0, T]
        self.T = 10
        self.td_x, self.td_Ham = self.model.solve(0, self.T, self.num_integrator, self.mu)

    def _get_training_data(self):
        assert hasattr(self, 'td_x'), 'Call _compute_solution() before calling _get_training_data()'
        data = self.td_x.all_data_to_numpy()

        # delete Dirichlet boundary values from training data
        data = np.delete(data, [0, self.n_x-1, self.n_x, 2*self.n_x-1], axis=1)

        T_training = 1
        n_training = int(T_training / self.dt)
        x = data[0:n_training,:]
        y = data[1:n_training+1,:] # shift by 1

        # convert to torch.Tensor
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x,y

    def _plot(self, epoch, prefix='', other_model=None, other_mu = None):
        this_model = self.model if other_model is None else other_model
        this_mu = self.mu if other_mu is None else other_mu

        this_td_Ham = self.td_Ham
        this_td_x = self.td_x
        if other_model is not None or other_mu is not None:
            this_td_x, this_td_Ham = this_model.solve(0, self.T, self.num_integrator, this_mu)

        x0 = this_model.initial_value(this_mu)
        td_x_surrogate = integrate(self.surrogate_model, x0.vec_q[1:-1], x0.vec_p[1:-1], 
            0, self.T, self.dt, custom_phase_space=Dirichlet1dNumpyPhaseSpace(self.dim))

        # Plot Hamiltonian
        ham_plt = plot_hamiltonian(this_td_Ham, td_x_surrogate, this_model, this_mu)
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
        x,y = self._get_training_data()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=1e-2)

        for epoch in range(self.epochs):
            print('training step: %d/%d' % (epoch, self.epochs))
    
            y1 = self.surrogate_model(x)
            loss = criterion(y1, y)
            optimizer.zero_grad()

            loss.backward()
            self.writer.add_scalar("Loss/train", loss, epoch)

            optimizer.step()

            if (epoch % 200) == 0:
                self._plot(epoch)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--model', choices=['standing_wave', 'transport', 'sine_gordon'], default='standing_wave')
    parser.add_argument('--dt', default=.1, type=float)
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
    args = parser.parse_args()

    expm = WaveExperiment(args)
    expm.run()