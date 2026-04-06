"""
maps.py — Discrete-time dynamical systems (maps).

Each concrete system defines:
    forward_map(x)  — the map  x_{n+1} = f(x_n)
    jac(x)          — analytical Jacobian  J(x) = ∂f/∂x  (vectorized over a trajectory)
"""

import numpy as np
from .base import DynamicalSystem
from .utils import njit, simulate_map
from .methods import discrete_qr_spectrum, discrete_qr_loop, discrete_qr_loop_2d


class DiscreteMap(DynamicalSystem):
    """Base class for discrete-time dynamical systems (maps)."""

    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, **kwargs)

    def compile(self) -> None:
        """Warm up JIT kernels for simulation and spectrum computation."""
        super().compile()
        x0 = np.ones(self.dim)
        self.simulate(x0, n_steps=1)
        qr_methods = ['householder', 'gram-schmidt'] if self.dim in [2, 3] else ['householder']
        for qm in qr_methods:
            self.discrete_qr_lyapunov_spectrum(qr_method=qm)

    def simulate(self, x0: np.ndarray, n_steps: int, n_burn: int = 0) -> np.ndarray:
        """
        Run the map forward for n_steps steps, discarding n_burn transients.

        Parameters
        ----------
        x0      : np.ndarray  — initial condition
        n_steps : int         — steps to record
        n_burn  : int         — steps to discard before recording (default 0)

        Returns
        -------
        x : np.ndarray, shape (n_steps, dim)
        """
        self.n_steps = n_steps
        x0_arr = np.atleast_1d(np.asarray(x0, dtype=float))
        self.x = simulate_map(self.forward_map, x0_arr, n_steps, n_burn, self.dim)
        return self.x

    def discrete_qr_lyapunov_spectrum(self, qr_method: str = 'householder') -> np.ndarray:
        """
        Compute the Lyapunov spectrum using the discrete-QR (Benettin) method.

        For 1-D maps this reduces to λ = ⟨log|f'(x)|⟩.
        For higher dimensions, iterative QR factorization of the Jacobian
        product separates growth rates.

        Parameters
        ----------
        qr_method : str — 'householder' (default) or 'gram-schmidt'

        Returns
        -------
        spectrum : np.ndarray, shape (dim,)
        """
        self.J = self.jac(self.x)

        if self.dim == 1:
            self.lyapunov_spectrum = np.array([np.mean(np.log(np.abs(self.J.flatten())))])
            return self.lyapunov_spectrum

        qr_func = self._get_qr_func(qr_method)

        if self.jit_enabled:
            if self.dim == 2:
                self.Q, self.R = discrete_qr_loop_2d(self.J, self.n_steps)
            else:
                self.Q, self.R = discrete_qr_loop(qr_func, self.J, self.n_steps, self.dim)
        else:
            Q = np.eye(self.dim)
            self.Q = self.R = np.empty((self.n_steps, self.dim, self.dim))
            for i in range(self.n_steps):
                Q, self.R[i] = qr_func(self.J[i] @ Q)
                self.Q[i] = Q

        self.lyapunov_spectrum = discrete_qr_spectrum(self.R)
        return self.lyapunov_spectrum


# ---------------------------------------------------------------------------
# Concrete maps
# ---------------------------------------------------------------------------

class LogisticMap(DiscreteMap):
    """
    1-D Logistic Map:  x_{n+1} = r·x_n·(1 - x_n)

    At r = 4 (default) the map is fully chaotic with λ = ln 2 ≈ 0.693.
    """

    def __init__(self, r: float = 4.0, **kwargs):
        self.r = r

        @njit
        def forward_map(x):
            return np.array([r * x[0] * (1.0 - x[0])])
        self.forward_map = forward_map

        super().__init__(dim=1, **kwargs)

    def jac(self, x: np.ndarray = None) -> np.ndarray:
        """Vectorized Jacobian over a trajectory. Returns shape (n, 1, 1)."""
        if x is None:
            x = self.x
        x = np.atleast_2d(x)
        return (self.r * (1.0 - 2.0 * x))[:, :, np.newaxis]


class HenonMap(DiscreteMap):
    """
    2-D Hénon Map:
        x_{n+1} = 1 - a·x_n² + y_n
        y_{n+1} = b·x_n

    Classical parameters (a, b) = (1.4, 0.3) produce the Hénon attractor.
    """

    def __init__(self, a: float = 1.4, b: float = 0.3, **kwargs):
        self.a = a
        self.b = b

        @njit
        def forward_map(x):
            return np.array([1.0 - a*x[0]**2 + x[1], b*x[0]])
        self.forward_map = forward_map

        super().__init__(dim=2, **kwargs)

    def jac(self, x: np.ndarray = None) -> np.ndarray:
        """Vectorized Jacobian over a trajectory. Returns shape (n, 2, 2)."""
        if x is None:
            x = self.x
        x = np.atleast_2d(x)
        n = x.shape[0]
        J = np.zeros((n, 2, 2))
        J[:, 0, 0] = -2.0 * self.a * x[:, 0]
        J[:, 0, 1] = 1.0
        J[:, 1, 0] = self.b
        return J