import matplotlib.pyplot as plt

def plot_Ham_sys(td_x, td_x_surrogate):
    fig = plt.figure(figsize=[10, 5])
    ax = plt.axes()
    # plot phase-space diagram
    ax.plot(list(td_x.all_vec_q()), list(td_x.all_vec_p()))
    ax.plot(list(td_x_surrogate.all_vec_q()), list(td_x_surrogate.all_vec_p()))
    ax.scatter(td_x._data[0].vec_q, td_x._data[0].vec_p, marker='o')
    ax.scatter(td_x._data[-1].vec_q, td_x._data[-1].vec_p, marker='s')
    ax.set_xlabel(r'$q$')
    ax.set_ylabel(r'$p$')
    ax.legend([
        'solution trajectory',
        'surrogate trajectory',
        'initial value',
        'final value'
    ])
    return fig

def plot_q(td_x, td_x_surrogate):
    fig = plt.figure(figsize=[15, 5])
    ax = plt.axes()

    ax.plot(list(td_x.all_t()), list(td_x.all_vec_q()), '.-')
    ax.plot(list(td_x_surrogate.all_t()), list(td_x_surrogate.all_vec_q()), '.-')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$q$')
    ax.legend([
        'solution trajectory',
        'surrogate trajectory'
    ])
    return fig

def plot_p(td_x, td_x_surrogate):
    fig = plt.figure(figsize=[15, 5])
    ax = plt.axes()

    ax.plot(list(td_x.all_t()), list(td_x.all_vec_p()), '.-')
    ax.plot(list(td_x_surrogate.all_t()), list(td_x_surrogate.all_vec_p()), '.-')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$p$')
    ax.legend([
        'solution trajectory',
        'surrogate trajectory'
    ])
    return fig

def plot_spatial_error(coord, loss):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel(r'$q0$')
    ax.set_ylabel(r'$p0$')
    s = ax.scatter(coord[:,0], coord[:,1], c=loss, s=20)
    fig.colorbar(s)
    return fig

def plot_hamiltonian(td_Ham, td_Ham_surrogate):
    fig = plt.figure(figsize=[15, 5])
    ax = plt.axes()

    ax.plot(td_Ham._t, td_Ham._data - td_Ham._data[0])
    ax.plot(td_Ham_surrogate._t, td_Ham_surrogate._data - td_Ham._data[0])
    ax.set_title('Hamiltonian vs. time')
    ax.set_xlabel(r'time $t$')
    ax.set_ylabel(r'Ham. $H(x(t,\mu), \mu) - H(x(t_0,\mu), \mu)$')
    ax.set_yscale('log')
    ax.legend([
        'solution Hamiltonian',
        'surrogate Hamiltonian'
    ])
    ax.relim()
    ax.autoscale_view()

    return fig