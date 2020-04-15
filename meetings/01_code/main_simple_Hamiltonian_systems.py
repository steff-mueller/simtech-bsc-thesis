# Symplectic Bachelor thesis
# Main script for simple TIME-INDEPENDENT Hamiltonian Systems

import numpy as np
import matplotlib.pyplot as plt
from simple_Hamiltonian_systems import HarmonicOscillator
from simple_Hamiltonian_systems import SimplePendulum

def plot_Ham_sys(X, t, all_Ham):
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    # plot phase-space diagram
    ax[0].plot(X[0,:], X[1,:])
    ax[0].scatter(X[0,0], X[1,0], marker='o')
    ax[0].scatter(X[0,-1], X[1,-1], marker='s')
    ax[0].legend([
        'solution trajectory',
        'initial value',
        'final value'
    ])
    ax[0].set_aspect('equal', 'box')
    ax[0].set_title('phase-space diagram')
    ax[0].set_xlabel(r'displacement $q(t)$')
    ax[0].set_ylabel(r'momentum $p(t)$')
    # plot Hamiltonian over time
    ax[1].plot(t, all_Ham - all_Ham[0])
    ax[1].set_title('Hamiltonian vs. time')
    ax[1].set_xlabel(r'time $t$')
    ax[1].set_ylabel(r'Ham. $H(x(t,\mu), \mu) - H(x(t_0,\mu), \mu)$')
    ax[1].set_yscale('log')
    plt.show()

if __name__ == "__main__":
    ## Harmonic oscillator
    model = HarmonicOscillator()
    mu = {'m': 1., 'k': 1., 'f': 0., 'q0': 1., 'p0': 0.}
    # compute solution for all t in [0, pi]
    X, t = model.solve(0, np.pi, np.pi/1e3, mu)
    all_Ham = model.Ham(X, mu)
    # plot solution
    plot_Ham_sys(X, t, all_Ham)

    ## Simple pendulum, swinging case
    model = SimplePendulum()
    mu = {'m': 1., 'g': 1., 'l': 1., 'q0': np.pi/2, 'p0': 0}
    # compute solution for all t in [0, 4]
    X, t = model.solve(0, 4, 4/2e3, mu)
    all_Ham = model.Ham(X, mu)
    # plot solution
    plot_Ham_sys(X, t, all_Ham)

    ## Simple pendulum, rotating case
    model = SimplePendulum()
    mu = {'m': 1., 'g': 1., 'l': 1., 'q0': np.pi, 'p0': 1.}
    # compute solution for all t in [0, 4]
    X, t = model.solve(0, 4, 4/2e3, mu)
    all_Ham = model.Ham(X, mu)
    # plot solution
    plot_Ham_sys(X, t, all_Ham)
