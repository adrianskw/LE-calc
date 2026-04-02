"""
maps.py — Discrete-time dynamical systems (maps).

Each system defines:
  - forward_map(x) : the map  x_{n+1} = f(x_n)
  - jac(x)         : the analytical Jacobian  J(x) = df/dx
"""

import numpy as np
from .base import DynamicalSystem
from .utils import njit, qr_GS_2x2, qr_GS_3x3, qr_HH
from .methods import discrete_qr_spectrum


# ===========================================================================
# JIT-Compiled Simulation Kernel (Stable & High-Performance)
# ===========================================================================

@njit
def _simulate_jit(map_func, x0: np.ndarray, n_steps: int, dim: int) -> np.ndarray:
    """Generic JIT-compiled simulation loop."""
    x = np.zeros((n_steps, dim))
    state = x0
    for i in range(n_steps):
        x[i] = state
        state = map_func(state)
    return x


@njit
def _discrete_qr_loop_2d_jit(J: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Highly optimized, fully-inlined 2x2 QR loop."""
    Q = np.eye(2)
    Q_out = np.zeros((n_steps, 2, 2))
    R_out = np.zeros((n_steps, 2, 2))
    for i in range(n_steps):
        # Inlined matmul and analytical 2x2 QR
        m00 = J[i, 0, 0]*Q[0, 0] + J[i, 0, 1]*Q[1, 0]
        m10 = J[i, 1, 0]*Q[0, 0] + J[i, 1, 1]*Q[1, 0]
        m01 = J[i, 0, 0]*Q[0, 1] + J[i, 0, 1]*Q[1, 1]
        m11 = J[i, 1, 0]*Q[0, 1] + J[i, 1, 1]*Q[1, 1]
        r11 = np.sqrt(m00*m00 + m10*m10)
        q00, q10 = m00 / r11, m10 / r11
        r12 = q00 * m01 + q10 * m11
        q01, q11 = -q10, q00
        r22 = q01 * m01 + q11 * m11
        Q[0, 0], Q[1, 0], Q[0, 1], Q[1, 1] = q00, q10, q01, q11
        Q_out[i] = Q
        R_out[i, 0, 0], R_out[i, 0, 1], R_out[i, 1, 1] = r11, r12, r22
    return Q_out, R_out


@njit
def _discrete_qr_loop_jit(qr_func, J: np.ndarray, n_steps: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Generic JIT-compiled QR re-orthonormalization loop."""
    Q = np.eye(dim)
    Q_out = np.zeros((n_steps, dim, dim))
    R_out = np.zeros((n_steps, dim, dim))
    for i in range(n_steps):
        M = J[i] @ Q
        Q_new, R = qr_func(M)
        Q = Q_new
        Q_out[i] = Q
        R_out[i] = R
    return Q_out, R_out


class DiscreteMap(DynamicalSystem):
    """
    Base class for discrete-time dynamical systems (maps).
    """

    def __init__(self, dim: int, eager_compile: bool = True):
        super().__init__(dim=dim, eager_compile=eager_compile)

    def jac(self, x: np.ndarray = None) -> np.ndarray:
        """Compute the analytical Jacobian J(x)."""
        raise NotImplementedError

    def _warmup_specific(self) -> None:
        """Trigger JIT compilation for simulation and spectrum calculation."""
        dummy_x = np.ones(self.dim)
        self.simulate(x0=dummy_x, n_steps=1)
        if self.jit_enabled:
            self.discrete_qr_lyapunov_spectrum(qr_method='householder')
            if self.dim in [2, 3]:
                self.discrete_qr_lyapunov_spectrum(qr_method='gram-schmidt')

    def simulate(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """Simulate the system for n_steps from x0."""
        self.n_steps = n_steps
        # stable one-line JIT bridge
        self.x = _simulate_jit(self.forward_map, np.atleast_1d(np.asarray(x0, dtype=float)), n_steps, self.dim)
        return self.x

    def discrete_qr_lyapunov_spectrum(self, qr_method: str = 'householder') -> np.ndarray:
        """Compute the Lyapunov spectrum using the discrete QR method."""
        self.J = self.jac(self.x)

        if self.dim == 1:
            self.lyapunov_spectrum = np.array([np.mean(np.log(np.abs(self.J.flatten())))])
            return self.lyapunov_spectrum

        qr_func =   qr_GS_2x2 if (qr_method == 'gram-schmidt' and self.dim == 2) else \
                    qr_GS_3x3 if (qr_method == 'gram-schmidt' and self.dim == 3) else \
                    qr_HH

        if self.jit_enabled:
            if self.dim == 2:
                self.Q, self.R = _discrete_qr_loop_2d_jit(self.J, self.n_steps)
            else:
                self.Q, self.R = _discrete_qr_loop_jit(qr_func, self.J, self.n_steps, self.dim)
            self.lyapunov_spectrum = discrete_qr_spectrum(self.R, 1.0)
            return self.lyapunov_spectrum

        Q = np.eye(self.dim)
        self.R = np.zeros((self.n_steps, self.dim, self.dim))
        self.Q = np.zeros((self.n_steps, self.dim, self.dim))
        for i in range(self.n_steps):
            Q, self.R[i] = qr_func(self.J[i] @ Q)
            self.Q[i] = Q

        self.lyapunov_spectrum = discrete_qr_spectrum(self.R, 1.0)
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