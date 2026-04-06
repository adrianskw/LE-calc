"""
odes.py — Continuous-time dynamical systems (ODEs).

Each concrete system defines:
    ode(x)    — vector field  f(x) = ẋ
    jac(x)    — analytical Jacobian  J(x) = ∂f/∂x  at state x
    xdot_H(x, xdot)  — contraction of velocity field with the Hessian,
                        used by the higher-order Taylor spectrum method

When Numba is available, @njit-compiled kernels are used throughout.
The method lookup tables in utils.py (RK_METHODS, RK_VAR_METHODS, QR_METHODS)
map string method names to their compiled handles.
"""

import numpy as np
from .base import DynamicalSystem
from .methods import (
    matrix_exponential_spectrum,
    taylor_spectrum,
    continuous_qr_spectrum,
    discrete_qr_spectrum,
)
from .utils import njit, RK_METHODS, RK_VAR_METHODS, simulate_ode, simulate_ode_var


class ODEs(DynamicalSystem):
    """Base class for continuous-time ODE systems."""

    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, **kwargs)

    def compile(self) -> None:
        """
        Warm up all JIT kernels needed by ODE systems:
        state steppers, variational steppers, and all spectrum methods.
        """
        super().compile()
        x0, Phi0 = np.ones(self.dim), np.eye(self.dim)
        dummy = np.array([Phi0])

        for qm in ['householder', 'gram-schmidt']:
            for m in ['RK2', 'RK4']:
                self.simulate(0.01, (0, 0.01), x0, method=m)
                self.simulate_var(0.01, (0, 0.01), x0, Phi0, method=m, qr_method=qm)
            matrix_exponential_spectrum(dummy, 0.01, qr_method=qm, order=1)
            taylor_spectrum(dummy, dummy, 0.01, qr_method=qm)
            continuous_qr_spectrum(dummy, dummy)
            discrete_qr_spectrum(dummy, 0.01)

    def _prepare_integration(self, dt: float, t_span: tuple[float, float],
                              method: str, is_var: bool = False):
        """
        Resolve the RK stepper and compute step counts from t_span.

        Parameters
        ----------
        dt      : float
        t_span  : (burn_in_time, end_time)
        method  : str   — 'RK2' or 'RK4'
        is_var  : bool  — True to use the variational (Phi) steppers

        Returns
        -------
        stepper_func : callable
        n_steps      : int  — recorded steps
        n_burn       : int  — discarded burn-in steps
        """
        lookup = RK_VAR_METHODS if is_var else RK_METHODS
        if method not in lookup:
            raise ValueError(f"Unsupported method '{method}'. Choose from {list(lookup)}.")
        t_burn, t_end = t_span
        self.n_steps = int((t_end - t_burn) / dt)
        return lookup[method], self.n_steps, int(t_burn / dt)

    def calc_xdot_H(self) -> np.ndarray:
        """
        Compute the Hessian-contraction matrices along the stored trajectory.

        H_i(x) = Σ_m f_m(x) · ∂²f_i/∂x_j ∂x_k   (indices j, k)

        This is the second-order correction used by the Taylor spectrum method.
        Must be called after simulate() has populated self.x.

        Returns
        -------
        xdot_H_history : np.ndarray, shape (n_steps, dim, dim)
        """
        self.xdot_H_history = np.empty((self.n_steps, self.dim, self.dim))
        for i in range(self.n_steps):
            xdot = self.ode(self.x[i])
            self.xdot_H_history[i] = self.xdot_H(self.x[i], xdot)
        return self.xdot_H_history

    def simulate(self, dt: float, t_span: tuple[float, float],
                 x0: np.ndarray, method: str = 'RK4') -> np.ndarray:
        """
        Integrate the ODE (state only) and store the trajectory in self.x.

        Parameters
        ----------
        dt     : float
        t_span : (burn_in_time, end_time) — burn-in is discarded
        x0     : np.ndarray — initial condition
        method : str        — 'RK2' or 'RK4' (default)

        Returns
        -------
        x : np.ndarray, shape (n_steps, dim)
        """
        step_func, n_steps, n_burn = self._prepare_integration(dt, t_span, method)
        self.x = simulate_ode(step_func, self.ode, dt, n_steps, n_burn,
                               np.asarray(x0, dtype=float), self.dim)
        return self.x

    def simulate_var(self, dt: float, t_span: tuple[float, float],
                     x0: np.ndarray, Phi0: np.ndarray,
                     method: str = 'RK4', qr_method: str = 'householder'):
        """
        Integrate state + variational equations with QR re-orthonormalization.

        Parameters
        ----------
        dt        : float
        t_span    : (burn_in_time, end_time)
        x0        : np.ndarray, shape (dim,)
        Phi0      : np.ndarray, shape (dim, dim) — initial basis frame (usually I)
        method    : str — integration solver, 'RK2' or 'RK4' (default)
        qr_method : str — 'householder' (default) or 'gram-schmidt'

        Returns
        -------
        x   : np.ndarray, shape (n_steps, dim)          — state trajectory
        phi : np.ndarray, shape (n_steps, dim, dim)     — pre-QR fundamental matrices
        Q   : np.ndarray, shape (n_steps, dim, dim)     — orthonormal frames
        R   : np.ndarray, shape (n_steps, dim, dim)     — upper-triangular growth factors
        J   : np.ndarray, shape (n_steps, dim, dim)     — Jacobians along trajectory
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
    Lorenz (1963) chaotic attractor.

        ẋ₁ = σ(x₂ - x₁)
        ẋ₂ = x₁(ρ - x₃) - x₂
        ẋ₃ = x₁x₂ - βx₃

    Classical parameters (σ, ρ, β) = (10, 28, 8/3) place the system in the
    chaotic regime with the characteristic butterfly attractor.
    """

    def __init__(self, sigma: float = 10.0, rho: float = 28.0,
                 beta: float = 8.0/3.0, **kwargs):
        self.sigma, self.rho, self.beta = sigma, rho, beta

        @njit
        def ode(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([sigma*(x2 - x1),
                             x1*(rho - x3) - x2,
                             x1*x2 - beta*x3])
        self.ode = ode

        @njit
        def jac(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([[-sigma,    sigma,  0.0 ],
                             [rho - x3, -1.0,   -x1  ],
                             [x2,        x1,    -beta]])
        self.jac = jac

        @njit
        def xdot_H(x, xdot):
            # Non-zero entries arise from the bilinear terms x₁x₃ and x₁x₂.
            res = np.zeros((3, 3))
            res[1, 0] = -xdot[2]
            res[1, 2] = -xdot[0]
            res[2, 0] =  xdot[1]
            res[2, 1] =  xdot[0]
            return res
        self.xdot_H = xdot_H

        super().__init__(dim=3, **kwargs)


class Rossler(ODEs):
    """
    Rössler chaotic attractor.

        ẋ₁ = -x₂ - x₃
        ẋ₂ =  x₁ + a·x₂
        ẋ₃ =  b + x₃(x₁ - c)

    Classical parameters (a, b, c) = (0.2, 0.2, 5.7) place the system in
    the chaotic regime with a single positive Lyapunov exponent.
    """

    def __init__(self, a: float = 0.2, b: float = 0.2,
                 c: float = 5.7, **kwargs):
        self.a, self.b, self.c = a, b, c

        @njit
        def ode(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([-x2 - x3,
                              x1 + a*x2,
                              b + x3*(x1 - c)])
        self.ode = ode

        @njit
        def jac(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([[0.0, -1.0, -1.0 ],
                             [1.0,  a,    0.0 ],
                             [x3,   0.0,  x1 - c]])
        self.jac = jac

        @njit
        def xdot_H(x, xdot):
            # Non-zero entries from the bilinear term x₁x₃.
            res = np.zeros((3, 3))
            res[2, 0] = xdot[2]
            res[2, 2] = xdot[0]
            return res
        self.xdot_H = xdot_H

        super().__init__(dim=3, **kwargs)