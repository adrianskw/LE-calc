"""
maps.py — Discrete-time dynamical systems (maps).

Each system defines:
  - forward_map(x) : the map  x_{n+1} = f(x_n)
  - jac(x)         : the analytical Jacobian  J(x) = df/dx
"""

import numpy as np
from .base import DynamicalSystem
from .utils import njit, qr_GS_2x2, qr_GS_3x3, qr_HH, simulate_map
from .methods import (
    discrete_qr_spectrum, 
    discrete_qr_loop, 
    discrete_qr_loop_2d
)


# ===========================================================================
# JIT-Compiled Simulation Kernel (Moved to utils.py)
# ===========================================================================


class DiscreteMap(DynamicalSystem):
    """
    Base class for discrete-time dynamical systems (maps).
    """

    def __init__(self, dim: int, eager_compile: bool = True):
        super().__init__(dim=dim, eager_compile=eager_compile)

    def _warmup_specific(self) -> None:
        """Trigger JIT compilation for simulation and spectrum calculation."""
        x0_dummy = np.ones(self.dim)
        self.simulate(x0_dummy, 1)
        for qm in (['householder', 'gram-schmidt'] if self.dim in [2, 3] else ['householder']):
            self.discrete_qr_lyapunov_spectrum(qm)

    def simulate(self, x0, n_steps: int, n_burn: int = 0):
        """Simulate the system for n_steps from x0, after burning n_burn steps."""
        self.n_steps, x0_arr = n_steps, np.atleast_1d(np.asarray(x0, dtype=float))
        self.x = simulate_map(self.forward_map, x0_arr, n_steps, n_burn, self.dim)
        return self.x

    def discrete_qr_lyapunov_spectrum(self, qr_method: str = 'householder') -> np.ndarray:
        """Compute the Lyapunov spectrum using the discrete QR method."""
        self.J = self.jac(self.x)
        
        # 1D case is a simple average of log-Jacobian
        if self.dim == 1:
            self.lyapunov_spectrum = np.array([np.mean(np.log(np.abs(self.J.flatten())))])
            return self.lyapunov_spectrum

        # Select the appropriate QR decomposition function
        if qr_method == 'gram-schmidt' and self.dim == 2:
            qr_func = qr_GS_2x2
        elif qr_method == 'gram-schmidt' and self.dim == 3:
            qr_func = qr_GS_3x3
        else:
            qr_func = qr_HH
        
        if self.jit_enabled:
            # Use specialized 2D loop for performance if applicable
            if self.dim == 2:
                self.Q, self.R = discrete_qr_loop_2d(self.J, self.n_steps)
            else:
                self.Q, self.R = discrete_qr_loop(qr_func, self.J, self.n_steps, self.dim)
        else:
            Q = np.eye(self.dim)
            self.Q = self.R = np.empty((self.n_steps, self.dim, self.dim))
            for i in range(self.n_steps):
                Q, self.R[i] = qrf(self.J[i] @ Q)
                self.Q[i] = Q

        self.lyapunov_spectrum = discrete_qr_spectrum(self.R)
        return self.lyapunov_spectrum


# ---------------------------------------------------------------------------
# Concrete maps
# ---------------------------------------------------------------------------

class LogisticMap(DiscreteMap):
    """1D Logistic Map: x_{n+1} = r * x_n * (1 - x_n)."""

    def __init__(self, r: float = 4.0, eager_compile: bool = True):
        self.r = r
        
        # Define Forward Map here:
        @njit
        def forward_map(x):
            return np.array([r * x[0] * (1.0 - x[0])])
        self.forward_map = forward_map
        
        # Super constructor handles warmup 
        super().__init__(dim=1, eager_compile=eager_compile)

    # vectorized jac method, no need for JIT compilation
    def jac(self, x: np.ndarray = None) -> np.ndarray:
        if x is None:
            x = self.x
        x = np.atleast_2d(x)
        res = self.r * (1.0 - 2.0 * x)
        return res[:, :, np.newaxis]


class HenonMap(DiscreteMap):
    """2D Hénon Map."""

    def __init__(self, a: float = 1.4, b: float = 0.3, eager_compile: bool = True):
        self.a = a
        self.b = b
        
        # Define Forward Map here:
        @njit
        def forward_map(x):
            return np.array([1.0 - a * x[0]**2 + x[1], b * x[0]])
        self.forward_map = forward_map
        
        # Super constructor handles warmup
        super().__init__(dim=2, eager_compile=eager_compile)

    # vectorized jac method, no need for JIT compilation
    def jac(self, x: np.ndarray = None) -> np.ndarray:
        if x is None:
            x = self.x
        x = np.atleast_2d(x)
        n = x.shape[0]
        J = np.zeros((n, 2, 2))
        J[:, 0, 0] = -2.0 * self.a * x[:, 0]
        J[:, 0, 1] = 1.0
        J[:, 1, 0] = self.b
        return J