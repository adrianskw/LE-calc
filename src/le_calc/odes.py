"""
odes.py — Continuous-time dynamical systems (ODEs).

Each system defines:
  - ode(x)  : the vector field  f(x) = dx/dt
  - jac(x)  : the analytical Jacobian  J(x) = df/dx  at a single 1-D state

When JIT is available (self.jit_enabled is True), @njit-compiled kernels are
used for the tight integration loops. The RK_METHODS / RK_VAR_METHODS lookup
tables in utils.py map method names to their compiled steppers.
"""

import numpy as np
from .base import DynamicalSystem
from .utils import (
    njit, qr_GS_2x2, qr_GS_3x3, qr_HH, 
    RK_METHODS, RK_VAR_METHODS,
    simulate_ode, simulate_ode_var
)


class ODEs(DynamicalSystem):
    """
    Base class for continuous-time dynamical systems (ODEs).
    Provides methods to define the vector field and Jacobian.
    """

    def __init__(self, dim: int, eager_compile: bool = True):
        super().__init__(dim=dim, eager_compile=eager_compile)

    def _warmup_specific(self) -> None:
        """Trigger JIT compilation for all RK methods and QR routines (condensed)."""
        from .methods import matrix_exponential_spectrum
        qr_m = ['householder', 'gram-schmidt']
        for m in ['RK2', 'RK4']:
            self.simulate(0.01, (0, 0.01), np.ones(self.dim), method=m)
            for qm in qr_m:
                self.simulate_var(0.01, (0, 0.01), np.ones(self.dim), np.eye(self.dim), m, qm)
                matrix_exponential_spectrum(np.array([np.eye(self.dim)]), 0.01, qr_method=qm)
            

    def _prepare_integration(self, dt: float, t_span: tuple[float, float], 
                               method: str, is_var: bool = False):
        """Standardized preparation for ODE integration."""
        # 1. Stepper Lookup
        lookup = RK_VAR_METHODS if is_var else RK_METHODS
        if method not in lookup:
            raise ValueError(f"Method '{method}' is not supported. "
                             f"Choose from {list(lookup)}.")
        
        step_func = lookup[method]

        # 2. Step Calculation
        t_burn, t_end = t_span
        n_burn = int(t_burn / dt)
        n_steps = int((t_end - t_burn) / dt)
        self.n_steps = n_steps

        return step_func, n_steps, n_burn

    def calc_jac(self) -> np.ndarray:
        """
        Compute and store the Jacobian matrices along the stored trajectory.

        Iterates over self.x, calling self.jac(state) for each 1D state and
        storing the result in self.J with shape (n_steps, dim, dim).

        Returns
        -------
        self.J : np.ndarray, shape (n_steps, dim, dim)
        """
        self.J = np.empty((self.n_steps, self.dim, self.dim))
        
        # Prefer JIT attributes 'ode'/'jac' over the class methods
        jac_func = self.jac
        
        for i in range(self.n_steps):
            self.J[i] = jac_func(self.x[i])
        return self.J

    def simulate(self, dt: float, t_span: tuple[float, float], x0: np.ndarray, method: str = 'RK4'):
        """Integrate the ODE system (state only) and store the trajectory."""
        step_func, n_steps, n_burn = self._prepare_integration(dt, t_span, method, is_var=False)
        self.x = simulate_ode(step_func, self.ode, dt, n_steps, n_burn, np.asarray(x0, dtype=float), self.dim)
        return self.x

    def simulate_var(self, dt: float, t_span: tuple[float, float], x0: np.ndarray, 
                     Phi0: np.ndarray, method: str = 'RK4', qr_method: str = 'householder'):
        """Integrate state + variational equations with QR re-orthonormalization."""
        step_func, n_steps, n_burn = self._prepare_integration(dt, t_span, method, is_var=True)
        
        # Select the appropriate QR decomposition function
        if qr_method == 'gram-schmidt' and self.dim == 2:
            qr_func = qr_GS_2x2
        elif qr_method == 'gram-schmidt' and self.dim == 3:
            qr_func = qr_GS_3x3
        else:
            qr_func = qr_HH

        self.x, self.phi, self.Q, self.R = simulate_ode_var(
            step_func, self.ode, self.jac, qr_func, dt, n_steps, n_burn, 
            np.asarray(x0, dtype=float), np.asarray(Phi0, dtype=float), self.dim
        )
        return self.x, self.phi, self.Q, self.R


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
                 beta: float = 8.0 / 3.0, eager_compile: bool = True):
        self.sigma = sigma
        self.rho   = rho
        self.beta  = beta

        # Define ODE and Jacobian here (same pattern as maps.py):
        s, r, b = self.sigma, self.rho, self.beta

        @njit
        def ode(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([s*(x2-x1), x1*(r-x3)-x2, x1*x2-b*x3])
        self.ode = ode

        @njit
        def jac(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([[-s,    s,   0.0],
                              [r-x3, -1.0, -x1],
                              [x2,   x1,  -b ]])
        self.jac = jac

        # Super constructor handles warmup 
        super().__init__(dim=3, eager_compile=eager_compile)


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
                 c: float = 5.7, eager_compile: bool = True):
        self.a = a
        self.b = b
        self.c = c

        # Define ODE and Jacobian here:
        a_p, b_p, c_p = self.a, self.b, self.c

        @njit
        def ode(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([-x2-x3, x1+a_p*x2, b_p+x3*(x1-c_p)])
        self.ode = ode

        @njit
        def jac(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([[0.0, -1.0, -1.0],
                             [1.0,  a_p,    0.0],
                             [x3,   0.0, x1-c_p]])
        self.jac = jac

        # Super constructor handles warmup
        super().__init__(dim=3, eager_compile=eager_compile)