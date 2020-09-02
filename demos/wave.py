import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.vectors import NumpyPhaseSpace
from models.wave import OscillatingModeLinearWaveProblem, FixedEndsLinearWaveProblem

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

        self.l = 1
        self._init_model(args.model)

        self.dim = 2*self.n_x-4 # -4 because Dirichlet boundary values removed
        self.surrogate_model = torch.nn.Sequential(
            UpperSymplecticConv1d(self.dim, bias=False),
            LowerSymplecticConv1d(self.dim, bias=False),
            UpperSymplecticConv1d(self.dim, bias=False)
        )

        if args.init_stormer_verlet:
            self._init_with_stormer_verlet()
    
    def _init_with_stormer_verlet(self):
        # initialize weights with Störmer-Verlet method
        self.surrogate_model[0].k.data = torch.tensor([0, self.dt/2, 0])
        self.surrogate_model[2].k.data = torch.tensor([0, self.dt/2, 0])

        dx = self.l/self.n_x
        factor = -self.dt*(self.mu['c']/dx)**2
        self.surrogate_model[1].k.data = torch.tensor([-1*factor, 2*factor, -1*factor])  

    def _init_model(self, model_name):
        if model_name == 'standing_wave':
            self.mu = {'c': .5}
            self.model = OscillatingModeLinearWaveProblem(self.l, self.n_x)
        elif model_name == 'transport':
            self.mu = {'c': .1, 'q0_supp': self.l/4}
            self.model = FixedEndsLinearWaveProblem(self.l, self.n_x)

    def _compute_solution(self): 
        # compute solution for all t in [0, T]
        self.T = 10
        self.td_x, self.td_Ham = self.model.solve(0, self.T, self.dt, self.mu)

    def _get_training_data(self):
        assert hasattr(self, 'td_x'), 'Call _compute_solution() before calling _get_training_data()'
        data = self.td_x.all_data_to_numpy()

        # delete Dirichlet boundary values from training data
        data = np.delete(data, [0, self.n_x-1, self.n_x, 2*self.n_x-1], axis=1)

        T_training = 4
        n_training = int(T_training / self.dt)
        x = data[0:n_training,:]
        y = data[1:n_training+1,:] # shift by 1

        # convert to torch.Tensor
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        y_scaled, self.scaler = scale_training_data(y)     
        return x,y_scaled

    def _plot(self, epoch, prefix='', other_model=None, other_mu = None):
        this_model = self.model if other_model is None else other_model
        this_mu = self.mu if other_mu is None else other_mu

        this_td_Ham = self.td_Ham
        this_td_x = self.td_x
        if other_model is not None or other_mu is not None:
            this_td_x, this_td_Ham = this_model.solve(0, self.T, self.dt, this_mu)

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
            y1_scaled = self.scaler(y1)
            loss = criterion(y1_scaled, y)        
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
    parser.add_argument('--model', choices=['standing_wave', 'transport'], default='standing_wave')
    parser.add_argument('--dt', default=.1, type=float)
    parser.add_argument('--nx', default=100, type=int)
    parser.add_argument('--print-params', default=False, nargs='?', const=True, type=bool)
    parser.add_argument('--post-plot-transport', default=False, nargs='?', const=True, type=bool, 
        help='Plot transport problem after training.')
    parser.add_argument('--init-stormer-verlet', default=False, nargs='?', const=True, type=bool, 
        help='Initialize weights with Stormer-Verlet integration scheme.')
    args = parser.parse_args()

    expm = WaveExperiment(args)
    expm.run()