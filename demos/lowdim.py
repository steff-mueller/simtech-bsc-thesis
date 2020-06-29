import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

from models.vectors import PhaseSpaceVectorList, NumpyPhaseSpace
from models.lowdim import SimplePendulum, HarmonicOscillator

from nn.training_data import generate_training_data
from nn.models import SympNet

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

def log_plot(model, surrogate_model, mu, q0, p0, dt, writer, epoch):
    td_x, td_Ham = model.solve(0, 4, dt, mu)
    td_x_surrogate = integrate(surrogate_model, q0, p0, 0, 4, dt)

    plt = plot_Ham_sys(td_x, td_x_surrogate)
    writer.add_figure('PhaseSpace', plt, epoch)

if __name__ == '__main__':
    # initialize TensorBoard writer
    writer = SummaryWriter(comment='lowdim')

    # model = HarmonicOscillator()
    # dt = 0.1
    # q0 = 1.
    # p0 = 0.
    # mu = {'m': 1., 'k': 1., 'f': 0., 'q0': q0, 'p0': p0}
    # x, y = generate_training_data(10000, model, mu, dt, 
    #   qmin = -2, qmax = 2,
    #   pmin = -2, pmax = 2)

    model = SimplePendulum()
    dt = 0.1
    q0 = np.pi/2
    p0 = 0.
    mu = {'m': 1., 'g': 1., 'l': 1., 'q0': q0, 'p0': p0}
    x, y = generate_training_data(10000, model, mu, dt, 
        qmin=-np.sqrt(2), qmax=np.sqrt(2),
        pmin=-np.pi/2, pmax=np.pi/2)

    surrogate_model = SympNet(layers = 8, sub_layers = 5, dim = 1, dt = 0.1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=1e-1)

    iter = 10000
    for epoch in range(iter):
       print('training step: %d/%d' % (epoch, iter))
       y1 = surrogate_model(x)
       loss = criterion(y1, y)
       writer.add_scalar("Loss/train", loss, epoch)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       if (epoch % 200) == 0:
           log_plot(model, surrogate_model, mu, q0, p0, dt, writer, epoch)

    log_plot(model, surrogate_model, mu, q0, p0, dt, writer, iter)  

    writer.add_graph(surrogate_model, torch.tensor([[q0, p0]]))

    writer.flush()