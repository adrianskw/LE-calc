import numpy as np
from .base import DynamicalSystem

class ODEs(DynamicalSystem):
    """
    Base class for continuous-time dynamical systems (ODEs).
    Provides methods to define the vector field and Jacobian.
    """
    
    def __init__(self, dim: int):
        super().__init__(dim=dim)

    def ode(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jac(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

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
        Supports 1D array of shape (3,) and 2D array of shape (N, 3).
        """
        if x.ndim == 1:
            x1, x2, x3 = x
            return np.array([
                [-self.sigma,   self.sigma,  0.0],
                [self.rho - x3, -1.0,       -x1],
                [x2,            x1,         -self.beta],
            ])
        else:
            x1, x2, x3 = x.T
            N = x.shape[0]
            J = np.zeros((N, 3, 3))
            J[:, 0, 0] = -self.sigma
            J[:, 0, 1] = self.sigma
            J[:, 1, 0] = self.rho - x3
            J[:, 1, 1] = -1.0
            J[:, 1, 2] = -x1
            J[:, 2, 0] = x2
            J[:, 2, 1] = x1
            J[:, 2, 2] = -self.beta
            return J


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
        if x.ndim == 1:
            x1, x2, x3 = x
            return np.array([
                [0.0, -1.0,     -1.0],
                [1.0, self.a,   0.0],
                [x3,  0.0,      x1 - self.c]
            ])
        else:
            x1, x2, x3 = x.T
            N = x.shape[0]
            J = np.zeros((N, 3, 3))
            J[:, 0, 1] = -1.0
            J[:, 0, 2] = -1.0
            J[:, 1, 0] = 1.0
            J[:, 1, 1] = self.a
            J[:, 2, 0] = x3
            J[:, 2, 2] = x1 - self.c
            return J