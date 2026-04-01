
import numpy as np
from le_calc.utils import integrate

class ODEs:
    """
    Base class for discrete-time dynamical systems (maps).
    Provides methods to simulate the trajectory and compute the Lyapunov spectrum.
    """
    
    def __init__(self, dim: int):
        self.dim: int = dim
        self.n_steps: int
        self.x: np.ndarray
        self.J: np.ndarray
        self.lyapunov_spectrum = np.empty(dim)

    def ode(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jac(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # def simulate(self, x0, n_steps):
    #     self.n_steps = n_steps
    #     self.x: np.ndarray = np.empty((self.n_steps, self.dim))

    #     x0 = np.atleast_1d(np.asarray(x0, dtype=float))   # handle scalars & arrays
    #     if len(x0) != self.dim:
    #         raise ValueError(
    #             f"{type(self).__name__} expects a {self.dim}-D initial condition, "
    #             f"but got x0 with length {len(x0)}."
    #         )
    #     self.x = np.zeros((self.n_steps, self.dim))
    #     for _ in range(min(1000, self.n_steps // 2)):            # warmup: discard transient
    #         x0 = self.forward_map(x0)
    #     for i in range(self.n_steps):
    #         self.x[i] = x0
    #         x0 = self.forward_map(x0)

    # def calc_lyapunov_spectrum(self):
    #     logAbsDiagR = np.zeros((self.n_steps, self.dim))
    #     Q = np.eye(self.dim)
    #     for i in range(self.n_steps):
    #         Q,R = np.linalg.qr(self.jac(self.x[i])@Q)
    #         logAbsDiagR[i] = np.log(np.abs(np.diag(R)))
    #     self.lyapunov_spectrum = np.mean(logAbsDiagR, axis=0)

class Lorenz63(ODEs):
    """
    The classic Lorenz 1963 system.

    Parameters
    ----------
    sigma : float
        Prandtl number (default: 10.0).
    rho : float
        Rayleigh number (default: 28.0).
    beta : float
        Geometric aspect ratio (default: 8/3).
    """

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
        super().__init__(dim=3)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def ode(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative dx/dt.
        """
        x1, x2, x3 = x
        return np.array([
            self.sigma * (x2 - x1),
            x1 * (self.rho - x3) - x2,
            x1 * x2 - self.beta * x3,
        ])

    def jac(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the analytical Jacobian matrix df/dx.
        """
        x1, x2, x3 = x
        return np.array([
            [-self.sigma,   self.sigma,  0.0],
            [self.rho - x3, -1.0,       -x1],
            [x2,            x1,         -self.beta],
        ])


class Rossler(ODEs):
    """
    The Rössler chaotic attractor.
    
    This system was designed to be one of the simplest possible chaotic 
    flow structures, governed by a single nonlinear term (z*x).
    
    Parameters
    ----------
    a : float (default: 0.2)
    b : float (default: 0.2)
    c : float (default: 5.7)
    """

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        super().__init__(dim=3)
        self.a = a
        self.b = b
        self.c = c

    def ode(self, x: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x
        return np.array([
            -x2 - x3,
            x1 + self.a * x2,
            self.b + x3 * (x1 - self.c)
        ])

    def jac(self, x: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x
        return np.array([
            [0.0, -1.0,     -1.0],
            [1.0, self.a,   0.0],
            [x3,  0.0,      x1 - self.c]
        ])