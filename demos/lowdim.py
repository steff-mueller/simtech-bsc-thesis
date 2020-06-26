import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

from models.vectors import PhaseSpaceVectorList, NumpyPhaseSpace
from models.lowdim import SimplePendulum

from nn.training_data import generate_training_data
from nn.models import SympNet

def train_model(model, criterion, optimizer, writer, iter):
   for epoch in range(iter):
       print('training step: %d/%d' % (epoch, iter))
       y1 = model(x)
       loss = criterion(y1, y)
       writer.add_scalar("Loss/train", loss, epoch)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

def integrate(model, q0, p0, t_start, t_end, dt, device=None):
    phase_space = NumpyPhaseSpace(2)
    result = PhaseSpaceVectorList()

    t_curr = t_start
    q_curr = q0
    p_curr = p0

    result.append(t_curr, phase_space.new_vector(q_curr, p_curr))
    
    while t_curr < t_end:
        y = model(torch.tensor([[q_curr, p_curr]]).to(device))

        t_curr += dt
        q_curr = y.data[0][0].item()
        p_curr = y.data[0][1].item()

        result.append(t_curr, phase_space.new_vector(q_curr, p_curr))  

    return result

def plot_Ham_sys(td_x, td_x_surrogate):
    fig = plt.figure()
    ax = plt.axes()
    # plot phase-space diagram
    ax.plot(list(td_x.all_vec_q()), list(td_x.all_vec_p()))
    ax.plot(list(td_x_surrogate.all_vec_q()), list(td_x_surrogate.all_vec_p()))
    ax.scatter(td_x._data[0].vec_q, td_x._data[0].vec_p, marker='o')
    ax.scatter(td_x._data[-1].vec_q, td_x._data[-1].vec_p, marker='s')
    ax.legend([
        'solution trajectory',
        'surrogate trajectory',
        'initial value',
        'final value'
    ])
    return fig

if __name__ == '__main__':
    # initialize TensorBoard writer
    writer = SummaryWriter(comment='lowdim')

    model = SimplePendulum()
    dt = 4/2e3
    q0 = np.pi/2
    p0 = 0.
    mu = {'m': 1., 'g': 1., 'l': 1., 'q0': q0, 'p0': p0}
    x, y = generate_training_data(10000, model, mu, dt, min = -np.pi, max = np.pi)

    surrogate_model = SympNet(layers = 8, sub_layers = 5, dim = 1, dt = 0.1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=1e-1)

    train_model(surrogate_model, criterion, optimizer, writer, 2000)

    td_x, td_Ham = model.solve(0, 4, dt, mu)
    td_x_surrogate = integrate(surrogate_model, q0, p0, 0, 4, dt)

    plt = plot_Ham_sys(td_x, td_x_surrogate)
    writer.add_figure('Example', plt)

    writer.flush()