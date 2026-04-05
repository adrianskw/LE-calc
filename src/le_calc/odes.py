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
from .methods import matrix_exponential_spectrum
from .utils import (
    njit, RK_METHODS, RK_VAR_METHODS,
    simulate_ode, simulate_ode_var
)


class ODEs(DynamicalSystem):
    """
    Base class for continuous-time dynamical systems (ODEs).
    Provides methods to define the vector field and Jacobian.
    """

    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, **kwargs)

    def compile(self) -> None:
        """Condensed JIT warmup: hit each stepper/QR routine once."""
        super().compile()
        x0, Phi0 = np.ones(self.dim), np.eye(self.dim)
        for m in ['RK2', 'RK4']:
            self.simulate(0.01, (0, 0.01), x0, method=m)
        for qm in ['householder', 'gram-schmidt']:
            self.simulate_var(0.01, (0, 0.01), x0, Phi0, 'RK4', qm)
            matrix_exponential_spectrum(np.array([Phi0]), 0.01, qr_method=qm)
            

    def _prepare_integration(self, dt: float, t_span: tuple[float, float], method: str, is_var: bool = False):
        """Prepare stepper, steps, and burn-in offset."""
        lookup = RK_VAR_METHODS if is_var else RK_METHODS
        if method not in lookup:
            raise ValueError(f"Method '{method}' unsupported.")
        t_burn, t_end = t_span
        self.n_steps = int((t_end - t_burn) / dt)
        return lookup[method], self.n_steps, int(t_burn / dt)


    def calc_xdot_H(self) -> np.ndarray:
        """
        Compute and store the pre-contracted Hessian xdot_H matrices along the stored trajectory.
        Vector field (xdot) is evaluated on-the-fly.

        Returns
        -------
        self.xdot_H_history : np.ndarray, shape (n_steps, dim, dim)
        """
        self.xdot_H_history = np.empty((self.n_steps, self.dim, self.dim))
        ode_func, xdot_H_func = self.ode, self.xdot_H
        for i in range(self.n_steps):
            xdot = ode_func(self.x[i])
            self.xdot_H_history[i] = xdot_H_func(self.x[i], xdot)
        return self.xdot_H_history

    def simulate(self, dt: float, t_span: tuple[float, float], x0: np.ndarray, method: str = 'RK4'):
        """Integrate the ODE system (state only) and store the trajectory."""
        step_func, n_steps, n_burn = self._prepare_integration(dt, t_span, method, is_var=False)
        self.x = simulate_ode(step_func, self.ode, dt, n_steps, n_burn, np.asarray(x0, dtype=float), self.dim)
        return self.x

    def simulate_var(self, dt: float, t_span: tuple[float, float], x0: np.ndarray, 
                     Phi0: np.ndarray, method: str = 'RK4', qr_method: str = 'householder'):
        """Integrate state + variational equations with QR re-orthonormalization."""
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
    The classic Lorenz 1963 system.

    Parameters
    ----------
    sigma : float  — Prandtl number (default 10.0)
    rho   : float  — Rayleigh number (default 28.0)
    beta  : float  — Geometric factor (default 8/3)
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

        # Super constructor handles warmup 
        super().__init__(dim=3, **kwargs)


class Rossler(ODEs):
    """
    The Rössler chaotic attractor.

    Parameters
    ----------
    a : float (default 0.2)
    b : float (default 0.2)
    c : float (default 5.7)
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

        # Super constructor handles warmup
        super().__init__(dim=3, **kwargs)