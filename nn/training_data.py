import torch
import numpy as np

def generate_training_data(n, model, mu, dt = 0.1, 
    qmin = -2, qmax = +2, 
    pmin = -2, pmax = +2,
    device=None):

    # do not change original dictionary
    mu = mu.copy()

    # TODO implmenet generic way to generate random values (in a subset)
    # generate random phase points in [-2,2]x[-2,2]
    rg = np.random.default_rng(seed=0)
    q = rg.uniform(qmin, qmax, size=(n,1))
    p = rg.uniform(pmin, pmax, size=(n,1))
    X_train = np.hstack((q,p))

    def time_step(x_train):
        # TODO implement generic way to set initial values
        mu['q0'] = x_train[0]
        mu['p0'] = x_train[1]

        # compute single time step h
        td_x, td_Ham = model.solve(0, dt+dt, dt, mu) # open time interval, therefore `dt+dt`
        t, x = td_x[1]
        return x.to_numpy()

    Y_train = np.apply_along_axis(time_step, 1, X_train)

    # convert numpy array to `torch.tensor`
    torch.set_default_dtype(torch.float64)
    X_train = torch.tensor(X_train, requires_grad=True).to(device)
    Y_train = torch.tensor(Y_train).to(device)

    return (X_train, Y_train)