import numpy as np

import torch
from torch import optim

import matplotlib.pyplot as plt

from livelossplot import PlotLosses

import sys
sys.path.insert(0, '../01_code')
from simple_Hamiltonian_systems import HarmonicOscillator

dt = 0.1 # time step

def generate_training_data(n=1000, device=None):
    model = HarmonicOscillator()

    # generate random phase points in [-2,2]x[-2,2]
    X_train = np.random.uniform(-2, 2, size=(n,2))

    def time_step(x_train):
        # compute one time step h
        mu = {'m': 1., 'k': 1., 'f': 0., 'q0': x_train[0], 'p0': x_train[1]}
        y_train, _ = model.solve(0, dt+dt, dt, mu) # open time interval, therefore `dt+dt`

        return y_train[:,1]

    Y_train = np.apply_along_axis(time_step, 1, X_train)

    # convert to `torch.tensor`
    torch.set_default_dtype(torch.float64)
    X_train = torch.tensor(X_train, requires_grad=True).to(device)
    Y_train = torch.tensor(Y_train).to(device)

    return (X_train, Y_train)

def train_fnn(loss_fn, X_train, Y_train, epochs=1000, liveloss=True, device=None):
    # train basic FNN
    nn_model = torch.nn.Sequential(
        torch.nn.Linear(2, 50),
        torch.nn.Sigmoid(),
        torch.nn.Linear(50, 50),
        torch.nn.Sigmoid(),
        torch.nn.Linear(50, 2)
    ).to(device)

    learning_rate = 1e-1

    opt = optim.Adam(nn_model.parameters(), lr=learning_rate)

    liveloss = PlotLosses()

    for t in range(epochs):
        loss = loss_fn(Y_train, nn_model, X_train)

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model.
        loss.backward()

        opt.step()
        opt.zero_grad()

        liveloss.update({'MSE loss': loss.item()})

        if (t % 25 and liveloss) == 0:
            liveloss.send()

    return nn_model

def integrate(model, q0 = 1., p0 = 0., t_start = 0., t_end = np.pi, device=None):
    t_curr = t_start
    q_curr = q0
    p_curr = p0

    t = np.array([t_curr])
    X = np.array([[q_curr, p_curr]])

    while t_curr < t_end:
        y = model(torch.tensor([[q_curr, p_curr]]).to(device))
        q_curr = y.data[0][0].item()
        p_curr = y.data[0][1].item()

        t = np.append(t, np.array([t_curr]), axis=0)
        X = np.append(X, np.array([[q_curr, p_curr]]), axis=0)

        t_curr += dt

    return (X,t)