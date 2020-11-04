import numpy as np
import matplotlib.pyplot as plt

def plot_q_over_time(td_x, td_x_surrogate, index):
    fig = plt.figure(figsize=[15, 5])
    ax = plt.axes()

    ax.plot(list(td_x.all_t()), list(item[index] for item in td_x.all_vec_q()), '.-')
    ax.plot(list(td_x_surrogate.all_t()), list(item[index] for item in td_x_surrogate.all_vec_q()), '.-')
    ax.legend([
        'solution trajectory',
        'surrogate trajectory'
    ])
    ax.set_title('q over time')
    ax.set_xlabel(r'time $t$')
    ax.set_ylabel(r'$q(t)$')
    return fig

def plot_p_over_time(td_x, td_x_surrogate, index):
    fig = plt.figure(figsize=[15, 5])
    ax = plt.axes()

    ax.plot(list(td_x.all_t()), list(item[index] for item in td_x.all_vec_p()), '.-')
    ax.plot(list(td_x_surrogate.all_t()), list(item[index] for item in td_x_surrogate.all_vec_p()), '.-')
    ax.legend([
        'solution trajectory',
        'surrogate trajectory'
    ])
    ax.set_title('p over time')
    ax.set_xlabel(r'time $t$')
    ax.set_ylabel(r'$p(t)$')
    return fig

def plot_q_over_domain(experiment, td_x, td_x_surrogate, t):
    fig = plt.figure(figsize=[15, 5])
    ax = plt.axes()

    coord = np.linspace(-experiment.l/2, experiment.l/2, experiment.n_x)
    _, x = td_x[int(t/experiment.dt)]
    _, x_surrogate = td_x_surrogate[int(t/experiment.dt)]

    ax.plot(coord, x.vec_q, '.-')
    ax.plot(coord, x_surrogate.vec_q, '.-')
    ax.legend([
        'solution trajectory',
        'surrogate trajectory'
    ])
    ax.set_title('q over domain')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$q(x)$')
    return fig

def plot_p_over_domain(experiment, td_x, td_x_surrogate, t):
    fig = plt.figure(figsize=[15, 5])
    ax = plt.axes()

    coord = np.linspace(-experiment.l/2, experiment.l/2, experiment.n_x)
    _, x = td_x[int(t/experiment.dt)]
    _, x_surrogate = td_x_surrogate[int(t/experiment.dt)]

    ax.plot(coord, x.vec_p, '.-')
    ax.plot(coord, x_surrogate.vec_p, '.-')
    ax.legend([
        'solution trajectory',
        'surrogate trajectory'
    ])
    ax.set_title('p over domain')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$p(x)$')
    return fig