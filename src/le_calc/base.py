import numpy as np
from .utils import HAS_NUMBA

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
    jit_enabled : bool
        Whether to use JIT-compiled function handles for optimization.
    """
    def __init__(self, dim: int, eager_compile: bool = True):
        self.dim, self.n_steps, self.jit_enabled = dim, 0, HAS_NUMBA
        self.x, self.lyapunov_spectrum = np.empty((0, dim)), np.empty(dim)
        self.J = self.phi = self.Q = self.R = np.empty((0, dim, dim))
        
        if self.jit_enabled and eager_compile:
            self.compile()

    def compile(self) -> None:
        """
        Triggers JIT compilation for all core handles by running dummy steps.
        This eliminates lazy-compilation delays during the first real simulation.
        """
        if not self.jit_enabled:
            return

        # 1. Warm up basic function handles (forward_map, ode, jac)
        for attr in ['forward_map', 'ode', 'jac']:
            if (handle := getattr(self, attr, None)) and callable(handle):
                try: handle(np.ones(self.dim))
                except: pass

        # 2. Trigger simulation loop compilation (subclass-specific)
        self._warmup_simulation()

    def _warmup_simulation(self) -> None:
        """Triggers a minimal 1-step simulation to warm up the integration/map loops."""
        self._warmup_specific()

    def _warmup_specific(self) -> None:
        """Subclass-specific warmup hook for complex kernels (e.g. simulation loops)."""
        pass