"""
odes.py — Continuous-time dynamical systems (ODEs).

Each system defines:
  - ode(x)  : the vector field  f(x) = dx/dt
  - jac(x)  : the analytical Jacobian  J(x) = df/dx  at a single 1-D state

When JIT is available (self.jit_enabled is True), @njit-compiled kernels are
used for the tight integration loops. The RK_METHODS, RK_VAR_METHODS, and 
QR_METHODS lookup tables in utils.py map method names to their compiled handles.
"""

import numpy as np
from .base import DynamicalSystem
from .methods import (
    matrix_exponential_spectrum, 
    taylor_spectrum, 
    continuous_qr_spectrum, 
    discrete_qr_spectrum
)
from .utils import (
    njit, RK_METHODS, RK_VAR_METHODS,
    simulate_ode, simulate_ode_var
)


class ODEs(DynamicalSystem):
    """
    Base class for continuous-time dynamical systems (ODEs).
    """

    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, **kwargs)

    def compile(self) -> None:
        """
        Condensed JIT warmup: hit each stepper/QR routine and Lyapunov method once.
        """
        super().compile()
        x0, Phi0 = np.ones(self.dim), np.eye(self.dim)
        dummy = np.array([Phi0])

        for qm in ['householder', 'gram-schmidt']:
            for m in ['RK2', 'RK4']:
                # 1. Warm up state and variational steppers
                self.simulate(0.01, (0, 0.01), x0, method=m)
                self.simulate_var(0.01, (0, 0.01), x0, Phi0, method=m, qr_method=qm)
            
            # 2. Warm up all Lyapunov spectrum methods
            matrix_exponential_spectrum(dummy, 0.01, qr_method=qm, order=1)
            taylor_spectrum(dummy, dummy, 0.01, qr_method=qm)
            continuous_qr_spectrum(dummy, dummy)
            discrete_qr_spectrum(dummy, 0.01)

    def _prepare_integration(self, dt: float, t_span: tuple[float, float], method: str, is_var: bool = False):
        """
        Internal helper to set up integration steps and burn-in times.

        Parameters
        ----------
        dt : float
        t_span : tuple[float, float]
        method : str
        is_var : bool

        Returns
        -------
        stepper_func, n_steps, n_burn : tuple
        """
        lookup = RK_VAR_METHODS if is_var else RK_METHODS
        if method not in lookup:
            raise ValueError(f"Method '{method}' unsupported.")
        t_burn, t_end = t_span
        self.n_steps = int((t_end - t_burn) / dt)
        return lookup[method], self.n_steps, int(t_burn / dt)

    def calc_xdot_H(self) -> np.ndarray:
        """
        Compute pre-contracted Hessian xdot_H matrices along the trajectory.

        Formula: [H_i]_jk = sum_m (f_i)_xm (f_m)_xjxk... 
        (More precisely, contraction of vector field with the local Hessian).

        Returns
        -------
        xdot_H_history : np.ndarray, shape (n_steps, dim, dim)
        """
        self.xdot_H_history = np.empty((self.n_steps, self.dim, self.dim))
        ode_func, xdot_H_func = self.ode, self.xdot_H
        for i in range(self.n_steps):
            xdot = ode_func(self.x[i])
            self.xdot_H_history[i] = xdot_H_func(self.x[i], xdot)
        return self.xdot_H_history

    def simulate(self, dt: float, t_span: tuple[float, float], x0: np.ndarray, method: str = 'RK4') -> np.ndarray:
        """
        Integrate the ODE system (state only) and store the trajectory.

        Parameters
        ----------
        dt : float
            Time step.
        t_span : tuple[float, float]
            Simulation range (burn_in_time, end_time).
        x0 : np.ndarray
            Initial condition.
        method : str, optional
            Numerical solver ('RK2' or 'RK4'). Defaults to 'RK4'.

        Returns
        -------
        x : np.ndarray, shape (n_steps, dim)
            The integrated trajectory.
        """
        step_func, n_steps, n_burn = self._prepare_integration(dt, t_span, method, is_var=False)
        self.x = simulate_ode(step_func, self.ode, dt, n_steps, n_burn, np.asarray(x0, dtype=float), self.dim)
        return self.x

    def simulate_var(self, dt: float, t_span: tuple[float, float], x0: np.ndarray, 
                     Phi0: np.ndarray, method: str = 'RK4', qr_method: str = 'householder'):
        """
        Simulate state + variational equations with QR re-orthonormalization.

        Parameters
        ----------
        dt : float
        t_span : tuple[float, float]
        x0 : np.ndarray
        Phi0 : np.ndarray
            Initial fundamental matrix (usually identity).
        method : str
            Integration solver ('RK2' or 'RK4').
        qr_method : str
            QR decomposition method ('householder' or 'gram-schmidt').

        Returns
        -------
        x : np.ndarray
            State trajectory history.
        phi : np.ndarray
            History of un-orthogonalized fundamental matrices.
        Q : np.ndarray
            History of orthogonalized basis frames.
        R : np.ndarray
            History of upper-triangular growth matrices (local contraction/expansion).
        J : np.ndarray
            History of analytical Jacobians evaluated along the path.
        """
        step_func, n_steps, n_burn = self._prepare_integration(dt, t_span, method, is_var=True)
        qr_func = self._get_qr_func(qr_method)

        self.x, self.phi, self.Q, self.R, self.J = simulate_ode_var(
            step_func, self.ode, self.jac, qr_func, dt, n_steps, n_burn, 
            np.asarray(x0, dtype=float), np.asarray(Phi0, dtype=float), self.dim
        )
        return self.x, self.phi, self.Q, self.R, self.J


# ---------------------------------------------------------------------------
# Concrete systems
# ---------------------------------------------------------------------------

class Lorenz63(ODEs):
    """
    Classic Lorenz 1963 chaotic attractor.
    """

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, 
                 beta: float = 8.0 / 3.0, **kwargs):
        self.sigma = sigma
        self.rho   = rho
        self.beta  = beta

        @njit
        def ode(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([sigma*(x2-x1), x1*(rho-x3)-x2, x1*x2-beta*x3])
        self.ode = ode

        @njit
        def jac(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([[-sigma,       sigma,  0.0],
                              [rho-x3, -1.0, -x1],
                              [x2,          x1,  -beta]])
        self.jac = jac

        @njit
        def xdot_H(x, xdot):
            res = np.zeros((3, 3))
            res[1, 0] = -xdot[2]
            res[1, 2] = -xdot[0]
            res[2, 0] = xdot[1]
            res[2, 1] = xdot[0]
            return res
        self.xdot_H = xdot_H

        super().__init__(dim=3, **kwargs)


class Rossler(ODEs):
    """
    Rössler chaotic attractor.
    """

    def __init__(self, a: float = 0.2, b: float = 0.2, 
                 c: float = 5.7, **kwargs):
        self.a = a
        self.b = b
        self.c = c

        @njit
        def ode(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([-x2-x3, x1+a*x2, b+x3*(x1-c)])
        self.ode = ode

        @njit
        def jac(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([[0.0, -1.0, -1.0],
                             [1.0,  a,    0.0],
                             [x3,   0.0, x1-c]])
        self.jac = jac

        @njit
        def xdot_H(x, xdot):
            res = np.zeros((3, 3))
            res[2, 0] = xdot[2]
            res[2, 2] = xdot[0]
            return res
        self.xdot_H = xdot_H

        super().__init__(dim=3, **kwargs)