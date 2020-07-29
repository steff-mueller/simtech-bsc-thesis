import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from models.wave import OscillatingModeLinearWaveProblem, FixedEndsLinearWaveProblem

from nn.models import SympNet, ConvLinearSympNet

from utils.plot2d import *

class WaveExperiment:
    def __init__(self, epochs, n_x):
        self.epochs = epochs
        
        self.writer = SummaryWriter(comment='wave')
        # TODO add parameters logging

        self.l = 1
        self.n_x = n_x
        self.model = OscillatingModeLinearWaveProblem(self.l, self.n_x)

        self.surrogate_model = ConvLinearSympNet(4, 2*self.n_x)
        #self.surrogate_model = SympNet(layers = 5, sub_layers = 4, dim = 2*self.n_x)

    def _compute_solution(self):
        self.mu = {'c': .5}
        # compute solution for all t in [0, T]
        self.T, self.dt = 10, .1
        self.td_x, self.td_Ham = self.model.solve(0, self.T, self.dt, self.mu)

    def _get_training_data(self):
        assert hasattr(self, 'td_x'), 'Call _compute_solution() before calling _get_training_data()'
        data = self.td_x.all_data_to_numpy()

        T_training = 4
        n_training = int(T_training / self.dt)
        x = data[0:n_training,:]
        y = data[1:n_training+1,:] # shift by 1

        # convert to torch.Tensor
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x,y

    def _plot(self, epoch):
        x0 = self.model.initial_value(self.mu)
        td_x_surrogate = self.surrogate_model.integrate(x0.vec_q, x0.vec_p, 
            0, self.T, self.dt)

        # Plot Hamiltonian
        ham_plt = plot_hamiltonian(self.td_Ham, td_x_surrogate, self.model, self.mu)
        self.writer.add_figure('/Hamiltonian', ham_plt, epoch) 

        # Plot q and p for left, middle and right knot
        # TODO
        
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
        self.writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--nx', default=100, type=int)
    args = parser.parse_args()

    expm = WaveExperiment(args.epochs, args.nx)
    expm.run()