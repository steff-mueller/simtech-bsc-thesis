# Symplectic Bachelor thesis
# Simple canonical, TIME-INDEPENDENT Hamiltonian Systems

from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import bmat, identity
from implicit_midpoint import implicit_midpoint

class CanonicalHamiltonianSystem(ABC):
    '''A parametric, canonical Hamiltonian system.

    Parameters
    ----------
    Ham
        The (parameter-dependent) Hamiltonian function. Has to be callable.
    gradHam
        The (parameter-dependent) gradient of the Hamiltonian. Has to be callable.
    initial_value
        The (parameter-dependent) initial value of the system. Has to be callable.
    dim
        The dimension of the Hamiltonian system.
    '''
    def __init__(self, Ham, gradHam, initial_value, dim):
        assert callable(Ham) and callable(gradHam) and callable(initial_value)
        self.Ham = Ham
        self.gradHam = gradHam
        self.initial_value = initial_value
        self.J = bmat([
            [None, identity(dim//2)],
            [-identity(dim//2), None]
        ])

    @abstractmethod
    def solve(self, t0, t1, dt, mu):
        '''Solve Hamiltonian system for all t in [t0, t1] with time-step width dt for the parameter vector mu.

        Parameters
        ----------
        t0
            The initial time.
        t1
            The final time.
        dt
            The (constant) time-step width.
        mu
            The parameter vector.
        '''
        pass

    def dxdt(self, x, mu):
        '''The RHS of the canonical Hamiltonian system:
            d/dt x(t, mu) = J * gradHam(x(t, mu), mu)

        Parameters
        ----------
        x
            The vector to evaluate the RHS for.
        mu
            The parameter vector.
        '''
        d = x.shape[0] // 2
        return self.J @ self.gradHam(x, mu)

class QuadraticHamiltonianSystem(CanonicalHamiltonianSystem):
    '''A |CanonicalHamiltonianSystem| with a quadratic Hamiltonina of the form.

        Ham(x, mu) = 1/2 x.T * H_op(mu) * x + x.T * h_op(mu)
        gradHam(x,mu) = H_op(mu) * x + h_op(mu) 
    '''
    def __init__(self, H_op, h_op, initial_value, dim):
        assert callable(H_op) and callable(h_op) and callable(initial_value)

        # rapidly evaluate a quadratic form
        # code from: https://stackoverflow.com/a/18542314
        Ham = lambda X, mu: 1/2 * np.einsum('...i,...i->...', X.T.dot(H_op(mu)), X.T) \
            + X.T.dot(h_op(mu))

        gradHam = lambda x, mu: H_op(mu) @ x + h_op(mu)

        super().__init__(Ham, gradHam, initial_value, dim)

class HarmonicOscillator(QuadraticHamiltonianSystem):
    '''A |QuadraticHamiltonianSystem| describing a harmonic oscillator.

        m d^2/d(t^2) q(t, mu) + k q(t, mu) = f(mu)

    The parameter vector consists of the parameter components:
        mu['m']: mass m
        mu['k']: spring stiffness k
        mu['f']: external force f (constant in time)
        mu['q0']: initial displacement
        mu['p0']: initial momentum
    '''
    def __init__(self):
        super().__init__(
            lambda mu: np.block([[mu['m'],0], [0, mu['k']]]),
            lambda mu: np.array([0, mu['f']]),
            lambda mu: np.array([mu['q0'], mu['p0']]),
            2
        )

    def solve(self, t0, t1, dt, mu):
        '''Implements explicit solution.
        '''
        assert isinstance(mu, dict)

        t = np.arange(t0, t1, dt)
        q0, p0 = self.initial_value(mu)
        omega0 = np.sqrt(mu['k']/mu['m'])

        X = np.array([
            q0*np.cos(omega0*(t-t0)) + p0*np.sin(omega0*(t-t0)),
            -mu['m']*omega0*q0*np.sin(omega0*(t-t0)) + p0*np.cos(omega0*(t-t0))
        ])

        return X, t

class SimplePendulum(CanonicalHamiltonianSystem):
    '''A |CanonicalHamiltonianSystem| describing the dynamics of a simple (mathematical) pendulum.

        m l d^2/(dt^2) q(t, mu) + m g sin(q(t, mu)) = 0

    The paramter vector consists of the parameter components
        mu['m']: mass m
        mu['g']: gravitational acceleration g
        mu['l']: pendulum length l
        mu['q0']: initial angle
        mu['p0']: initial momentum
    '''
    def __init__(self):
        super().__init__(
            lambda X, mu: X[1,:]**2/(2*mu['m']*mu['l']**2) + mu['m']*mu['g']*mu['l']*(1-np.cos(X[0,:])),
            lambda x, mu: np.array([
                mu['m']*mu['g']*mu['l']*np.sin(x[0]),
                x[1]/(mu['m']*mu['l']**2)
            ]),
            lambda mu: np.array([mu['q0'], mu['p0']]),
            2
        )

        # Hessian of Ham
        self.DxGradHam = lambda x, mu: np.block([
            [mu['m']*mu['g']*mu['l']*np.cos(x[1]),                      0],
            [                                   0, 1/(mu['m']*mu['l']**2)]
        ])
        # Jacobian of dxdt (to solve NLP in integrator with Newton's method)
        self.dxdt_jac = lambda x, mu: self.J @ self.DxGradHam(x, mu)

    def solve(self, t0, t1, dt, mu):
        '''Approximates with implicit midpoint integrator.
        '''
        assert isinstance(mu, dict)
        
        t = np.arange(t0, t1, dt)
        m = mu['m']
        g = mu['g']
        l = mu['l']
        x0 = self.initial_value(mu)

        return implicit_midpoint(
            lambda _,x: self.dxdt(x, mu),
            lambda _,x: self.dxdt_jac(x, mu),
            [t0, t1],
            dt,
            x0
        )

        