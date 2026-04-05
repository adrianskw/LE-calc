import numpy as np
from .utils import HAS_NUMBA, QR_METHODS

class DynamicalSystem:
    """
    Base class for all dynamical systems (ODEs and Maps).

    Attributes
    ----------
    dim : int
        Dimension of the system state space.
    n_steps : int
        Number of steps simulated or integrated.
    x : np.ndarray
        Trajectory of the system, shape (n_steps, dim).
    J : np.ndarray
        Jacobian matrices along the trajectory, shape (n_steps, dim, dim).
    phi : np.ndarray
        Calculated fundamental matrices, shape (n_steps, dim, dim).
    Q : np.ndarray
        Orthogonal matrices from QR decomposition, shape (n_steps, dim, dim).
    R : np.ndarray
        Upper triangular matrices from QR decomposition, shape (n_steps, dim, dim).
    lyapunov_spectrum : np.ndarray
        Calculated Lyapunov exponents, shape (dim,).
    xdot_H_history : np.ndarray
        Directly computed contraction of vector field and Hessian, shape (n_steps, dim, dim).
    jit_enabled : bool
        Whether to use JIT-compiled function handles for optimization.
    """
    def __init__(self, dim: int, **kwargs):
        eager_compile = kwargs.get('eager_compile', True)
        self.dim, self.n_steps, self.jit_enabled = dim, 0, HAS_NUMBA
        self.x, self.lyapunov_spectrum = np.empty((0, dim)), np.empty(dim)
        self.J = self.phi = self.Q = self.R = np.empty((0, dim, dim))
        self.xdot_H_history = np.empty((0, dim, dim))
        
        if self.jit_enabled and eager_compile:
            self.compile()

    def _get_qr_func(self, qr_method: str):
        """Standardized QR function lookup."""
        if qr_method == 'gram-schmidt':
            return QR_METHODS[f'gram-schmidt-{self.dim}x{self.dim}']
        return QR_METHODS['householder']

    def compile(self) -> None:
        """
        Triggers JIT compilation for all core handles by running dummy steps.
        Subclasses should override this and call super().compile() for full coverage.
        """
        if not self.jit_enabled:
            return

        # 1. Warm up basic function handles (forward_map, ode, jac, xdot_H)
        ones = np.ones(self.dim)
        for attr in ['forward_map', 'ode', 'jac', 'xdot_H']:
            if (handle := getattr(self, attr, None)) and callable(handle):
                try: 
                    handle(ones, ones) if attr == 'xdot_H' else handle(ones)
                except Exception: pass