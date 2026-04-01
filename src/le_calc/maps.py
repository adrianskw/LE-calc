import numpy as np
from .base import DynamicalSystem
from .utils import qr_2x2, qr_3x3, discrete_qr_spectrum

class DiscreteMap(DynamicalSystem):
    """
    Base class for discrete-time dynamical systems (maps).
    Provides methods to simulate the trajectory and compute the Lyapunov spectrum.
    """
    
    def __init__(self, dim: int):
        super().__init__(dim=dim)

    def forward_map(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jac(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def simulate(self, x0: np.ndarray, n_steps: int):
        """
        Simulate the system for n_steps from x0.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial condition.
        n_steps : int
            Number of iterations.
        """
        self.n_steps = n_steps
        self.x = np.zeros((n_steps, self.dim))
        state = np.atleast_1d(np.asarray(x0, dtype=float))
        for i in range(n_steps):
            self.x[i] = state
            state = self.forward_map(state)

    def discrete_qr_lyapunov_spectrum(self, qr_method: str = 'householder'):
        """
        Compute the Lyapunov spectrum using the discrete QR method.
        
        Parameters
        ----------
        qr_method : str
            QR decomposition method. 'householder' (default) or 'gram-schmidt'.

        Returns
        -------
        spectrum : np.ndarray
            The calculated Lyapunov exponents.
        """
        
        # Pre-compute the Jacobians along the trajectory.
        self.J = self.jac(self.x)
        
        # 1D Specialization: Extremely fast vectorized log-sum
        if self.dim == 1:
            log_abs_j = np.log(np.abs(self.J.flatten()))
            self.lyapunov_spectrum = np.array([np.mean(log_abs_j)])
            return self.lyapunov_spectrum

        # If requested, use fast analytical QR routines for small dimensions
        R_history = np.zeros((self.n_steps, self.dim, self.dim))
        Q = np.eye(self.dim)
        
        if qr_method == 'gram-schmidt' and self.dim in [2, 3]:
            qr_func = qr_2x2 if self.dim == 2 else qr_3x3
            for i in range(self.n_steps):
                Q, R_history[i] = qr_func(self.J[i] @ Q)
        else:
            for i in range(self.n_steps):
                Q, R_history[i] = np.linalg.qr(self.J[i] @ Q)
        
        self.lyapunov_spectrum = discrete_qr_spectrum(R_history, 1.0) # Maps have dt=1.0 implicitly
        return self.lyapunov_spectrum


class LogisticMap(DiscreteMap):
    """
    1D Logistic Map: x_{n+1} = r * x_n * (1 - x_n).
    """

    def __init__(self, r: float = 4.0):
        super().__init__(dim=1)
        self.r = r

    def forward_map(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.r * x[0] * (1.0 - x[0])])

    def jac(self, x: np.ndarray) -> np.ndarray:
        # Returns derivatives f'(x) = r*(1 - 2x)
        # If x is 1D (scalar-like) shape (1,), returns (1, 1)
        # If x is 2D shape (N, 1), returns (N, 1, 1)
        if x.ndim == 1:
            return np.array([[self.r * (1.0 - 2.0 * x[0])]])
        else:
            N = x.shape[0]
            J = (self.r * (1.0 - 2.0 * x)).reshape(N, 1, 1)
            return J


class HenonMap(DiscreteMap):
    """
    2D Henon Map: 
    x_{n+1} = 1 - a*x_n^2 + y_n
    y_{n+1} = b*x_n
    """

    def __init__(self, a: float = 1.4, b: float = 0.3):
        super().__init__(dim=2)
        self.a = a
        self.b = b

    def forward_map(self, x: np.ndarray) -> np.ndarray:
        x_next = 1.0 - self.a * x[0]**2 + x[1]
        y_next = self.b * x[0]
        return np.array([x_next, y_next])

    def jac(self, x: np.ndarray) -> np.ndarray:
        # J = [[-2*a*x, 1], [b, 0]]
        if x.ndim == 1:
            return np.array([
                [-2.0 * self.a * x[0], 1.0],
                [self.b,                0.0]
            ])
        else:
            N = x.shape[0]
            J = np.zeros((N, 2, 2))
            J[:, 0, 0] = -2.0 * self.a * x[:, 0]
            J[:, 0, 1] = 1.0
            J[:, 1, 0] = self.b
            return J
