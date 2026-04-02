"""
maps.py — Discrete-time dynamical systems (maps).

Each system defines:
  - forward_map(x) : the map  x_{n+1} = f(x_n)
  - jac(x)         : the analytical Jacobian  J(x) = df/dx  at a single 1-D state

When JIT is available (self.jit_enabled is True), @njit-compiled closures are built
at construction time and used in the simulation and Jacobian loops. When JIT is
disabled, the standard Python methods are used directly.
"""

import numpy as np
from .base import DynamicalSystem
from .utils import njit, qr_2x2, qr_3x3
from .methods import discrete_qr_spectrum


class DiscreteMap(DynamicalSystem):
    """
    Base class for discrete-time dynamical systems (maps).
    """

    def __init__(self, dim: int, eager_compile: bool = True):
        super().__init__(dim=dim, eager_compile=eager_compile)

    def forward_map(self, x: np.ndarray) -> np.ndarray:
        """Compute x_{n+1} = f(x_n)."""
        raise NotImplementedError

    def jac(self, x: np.ndarray = None) -> np.ndarray:
        """
        Compute the analytical Jacobian J(x) = df/dx at a single state or batch.

        Parameters
        ----------
        x : np.ndarray, optional
            The state(s) at which to compute the Jacobian. 
            If None (default), uses the stored trajectory self.x.

        Returns
        -------
        J : np.ndarray
            Jacobian matrix (dim, dim) or batch of matrices (n, dim, dim).
        """
        raise NotImplementedError

    def _get_map_handle(self):
        """
        Helper to select the correct map handle based on JIT status.
        """
        if self.jit_enabled and hasattr(self, '_forward_jit'):
            return self._forward_jit
        return self.forward_map

    def simulate(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Simulate the system for n_steps from x0.

        Parameters
        ----------
        x0 : np.ndarray   — Initial condition.
        n_steps : int     — Number of iterations.
        """
        self.n_steps = n_steps
        self.x = np.zeros((n_steps, self.dim))
        state = np.atleast_1d(np.asarray(x0, dtype=float))

        map_func = self._get_map_handle()

        for i in range(n_steps):
            self.x[i] = state
            state = map_func(state)
        return self.x

    def discrete_qr_lyapunov_spectrum(self, qr_method: str = 'householder') -> np.ndarray:
        """
        Compute the Lyapunov spectrum using the discrete QR method.

        Parameters
        ----------
        qr_method : str
            'householder' (default) or 'gram-schmidt'.

        Returns
        -------
        spectrum : np.ndarray, shape (dim,)
        """
        self.J = self.jac(self.x)

        # 1-D fast path: vectorised log-sum
        if self.dim == 1:
            self.lyapunov_spectrum = np.array([np.mean(np.log(np.abs(self.J.flatten())))])
            return self.lyapunov_spectrum

        Q = np.eye(self.dim)
        self.R = np.zeros((self.n_steps, self.dim, self.dim))
        self.Q = np.zeros((self.n_steps, self.dim, self.dim))

        if qr_method == 'gram-schmidt' and self.dim == 2:
            qr_func = qr_2x2
        elif qr_method == 'gram-schmidt' and self.dim == 3:
            qr_func = qr_3x3
        else:
            qr_func = np.linalg.qr

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
        super().__init__(dim=1, eager_compile=eager_compile)

    def _setup_jit_functions(self):
        """Build @njit-compiled closures with baked-in parameters."""
        r = self.r

        @njit
        def forward_jit(x):
            return np.array([r * x[0] * (1.0 - x[0])])

        @njit
        def jac_jit(x):
            return np.array([[r * (1.0 - 2.0 * x[0])]])

        self._forward_jit = forward_jit
        self._jac_jit     = jac_jit

    def forward_map(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.r * x[0] * (1.0 - x[0])])

    def jac(self, x: np.ndarray = None) -> np.ndarray:
        if x is None:
            x = self.x
        x = np.asarray(x)
        if x.ndim == 1:
            return np.array([[self.r * (1.0 - 2.0 * x[0])]])
        else:
            # Batch case: (n, 1) -> (n, 1, 1)
            res = self.r * (1.0 - 2.0 * x)
            return res[:, :, np.newaxis]


class HenonMap(DiscreteMap):
    """
    2D Hénon Map:
      x_{n+1} = 1 - a * x_n^2 + y_n
      y_{n+1} = b * x_n
    """

    def __init__(self, a: float = 1.4, b: float = 0.3, eager_compile: bool = True):
        self.a = a
        self.b = b
        super().__init__(dim=2, eager_compile=eager_compile)

    def _setup_jit_functions(self):
        """Build @njit-compiled closures with baked-in parameters."""
        a, b = self.a, self.b

        @njit
        def forward_jit(x):
            return np.array([1.0 - a * x[0]**2 + x[1], b * x[0]])

        @njit
        def jac_jit(x):
            return np.array([[-2.0 * a * x[0], 1.0],
                              [b,               0.0]])

        self._forward_jit = forward_jit
        self._jac_jit     = jac_jit

    def forward_map(self, x: np.ndarray) -> np.ndarray:
        return np.array([1.0 - self.a * x[0]**2 + x[1], self.b * x[0]])

    def jac(self, x: np.ndarray = None) -> np.ndarray:
        if x is None:
            x = self.x
        x = np.asarray(x)
        if x.ndim == 1:
            return np.array([[-2.0 * self.a * x[0], 1.0],
                             [self.b,               0.0]])
        else:
            # Batch case: (n, 2) -> (n, 2, 2)
            n = x.shape[0]
            J = np.zeros((n, 2, 2))
            J[:, 0, 0] = -2.0 * self.a * x[:, 0]
            J[:, 0, 1] = 1.0
            J[:, 1, 0] = self.b
            return J
