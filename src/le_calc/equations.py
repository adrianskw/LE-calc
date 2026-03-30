"""
equations.py — Dynamical Systems for Lyapunov exponent calculations.

Provides classes and functions for various known dynamical systems (ODEs and discrete maps),
allowing users to calculate the state derivative `f` and the Jacobian `jac` separately.
"""

import numpy as np

class Lorenz63:
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
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def f(self, t: float, x: np.ndarray) -> list[float]:
        """
        Compute the derivative dx/dt.
        """
        x1, x2, x3 = x
        return [
            self.sigma * (x2 - x1),
            x1 * (self.rho - x3) - x2,
            x1 * x2 - self.beta * x3,
        ]

    def jac(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the analytical Jacobian matrix df/dx.
        """
        x1, x2, x3 = x
        return np.array([
            [-self.sigma,   self.sigma,        0.0],
            [self.rho - x3, -1.0,           -x1],
            [x2,            x1,             -self.beta],
        ])


class Rossler:
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
        self.a = a
        self.b = b
        self.c = c

    def f(self, t: float, x: np.ndarray) -> list[float]:
        x1, x2, x3 = x
        return [
            -x2 - x3,
            x1 + self.a * x2,
            self.b + x3 * (x1 - self.c)
        ]

    def jac(self, t: float, x: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x
        return np.array([
            [0.0, -1.0,     -1.0],
            [1.0, self.a,   0.0],
            [x3,  0.0,      x1 - self.c]
        ])


class Lorenz96:
    """
    The Lorenz-96 (L96) atmospheric model.
    A continuous N-dimensional spatiotemporal chaotic system.
    
    Parameters
    ----------
    N : int
        Number of spatial grid coordinates (default: 40).
    F : float
        External forcing constant (default: 8.0).
    """

    def __init__(self, N: int = 40, F: float = 8.0):
        self.N = N
        self.F = F

    def f(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative dx/dt.
        """
        x = np.asarray(x)
        # We use fast numpy rolling to calculate dx_i / dt across all N dimensions 
        # instead of a slow python for-loop.
        # equation: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + self.F

    def jac(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the analytical Jacobian matrix df/dx.
        Because dimensions scale to N=40 (or larger), we populate a matrix programmatically.
        """
        J = np.zeros((self.N, self.N))
        
        # In a very large N simulation this could be further vectorized,
        # but for N <= ~1000 a simple localized loop is perfectly fast.
        for i in range(self.N):
            J[i, i] = -1.0
            
            # The indices "wrap around" periodically
            i_plus_1 = (i + 1) % self.N
            i_minus_1 = (i - 1) % self.N
            i_minus_2 = (i - 2) % self.N
            
            # Derivative w.r.t x_{i+1}
            J[i, i_plus_1] = x[i_minus_1]
            
            # Derivative w.r.t x_{i-1}
            J[i, i_minus_1] = x[i_plus_1] - x[i_minus_2]
            
            # Derivative w.r.t x_{i-2}
            J[i, i_minus_2] = -x[i_minus_1]
            
        return J


class LogisticMap:
    """
    The classic Logistic map: x_{n+1} = r * x_n * (1 - x_n).
    A highly studied 1D discrete map demonstrating period-doubling bifurcations
    and chaos.

    Parameters
    ----------
    r : float
        Population growth rate parameter (default: 4). Chaotic regime is > ~3.57.
    """

    def __init__(self, r: float = 4.0):
        self.r = r

    def f(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the next mapped value x_{n+1}.
        (Argument `t` is included for signature consistency with ODEs, but is unused.)
        """
        x = np.asarray(x)
        return self.r * x * (1.0 - x)

    def jac(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the analytical Jacobian matrix (1x1).
        """
        x = np.asarray(x)
        return np.array([[self.r * (1.0 - 2.0 * x.item())]])


class HenonMap:
    """
    The Hénon map. A 2D discrete-time dynamical system that exhibits chaotic behavior.
    
    Parameters
    ----------
    a : float (default: 1.4)
    b : float (default: 0.3)
    """

    def __init__(self, a: float = 1.4, b: float = 0.3):
        self.a = a
        self.b = b

    def f(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the next mapped state.
        """
        x1, x2 = x
        return np.array([
            1.0 - self.a * x1**2 + x2,
            self.b * x1
        ])

    def jac(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the analytical Jacobian matrix.
        """
        x1, _ = x
        return np.array([
            [-2.0 * self.a * x1, 1.0],
            [self.b,             0.0]
        ])
