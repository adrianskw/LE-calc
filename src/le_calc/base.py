import numpy as np

class DynamicalSystem:
    """
    Base class for all dynamical systems (ODEs and Maps).
    """
    def __init__(self, dim: int):
        self.dim: int = dim
        self.n_steps: int = 0
        self.x: np.ndarray = np.empty((0, dim))
        self.J: np.ndarray = np.empty((0, dim, dim))
        self.lyapunov_spectrum = np.empty(dim)
