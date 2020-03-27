# Symplectic Bachelor thesis
# Implicit midpoint integrator

import numpy as np
from scipy.sparse import identity

def implicit_midpoint(ode, jac, tspan, dt, x0, verbose=True, tol=1e-12):
    '''The implicit midpoint rule for non-linear systems.

    Parameters
    ----------
    ode
        The RHS of the ode as function of t and x.
    jac
        The derivative of the RHS w.r.t. x as function of t and x.
    tspan
        A tuple of length two containing the inital time t0 and the final time t1.
    x0
        The initial value x0.
    verbose
        A flag wheter to print status.
    tol
        Tolerance for the Newton algorithm.
    '''
    t = np.arange(tspan[0], tspan[1], dt)
    N = len(t)
    At = lambda x,y: (x+y)/2
    n_dof = x0.shape[0]
    X = np.zeros([n_dof, N])
    X[:,0] = x0

    # compute solution for every time step
    for n in range(0,N-1):
        t_mid  = At(t[n],t[n+1])
        x_new  = X[:,n]
        x_old  = X[:,n]
        res, J = _implicit_midpoint_iteration(x_new, x_old, t_mid, dt, ode, jac)
        it_new = 1

        # apply Newton's method to find solution of current time step
        while np.linalg.norm(res) > tol:
            x_new  = x_new - np.linalg.solve(J, res)
            res, J = _implicit_midpoint_iteration(x_new, x_old, t_mid, dt, ode, jac)
            it_new = it_new + 1
        X[:, n+1] = x_new

        # Print progress in multiples of 10%
        if verbose and np.floor(((n-1)/N)*10) < np.floor((n/N)*10):
            print ('implicit midpoint: %3d%%' % (np.floor((n/N)*10)*10))
            print ('implicit midpoint step: %d/%d, Newton iterations: %d, residuum norm: %5.4e' % (n, N, it_new, np.linalg.norm(res)))
    
    return X, t


def _implicit_midpoint_iteration(y_new, y_old, t_mid, dt, ode, jac):
    '''One Newton step.
    '''
    y_mid  = (y_new + y_old)/2
    F      = y_new - y_old - dt * ode(t_mid, y_mid)
    J      = identity(len(y_new)) - dt/2*jac(t_mid, y_mid)
    return F, J
