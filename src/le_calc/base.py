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
    lyapunov_spectrum : np.ndarray
        Calculated Lyapunov exponents, shape (dim,).
    jit_enabled : bool
        Whether to use JIT-compiled function handles for optimization.
    """
    def __init__(self, dim: int, eager_compile: bool = True):
        self.dim: int = dim
        self.n_steps: int = 0
        self.x: np.ndarray = np.empty((0, dim))
        self.J: np.ndarray = np.empty((0, dim, dim))
        self.phi: np.ndarray = np.empty((0, dim, dim))
        self.Q: np.ndarray = np.empty((0, dim, dim))
        self.R: np.ndarray = np.empty((0, dim, dim))
        self.lyapunov_spectrum = np.empty(dim)
        
        # JIT is enabled by default if numba is installed
        self.jit_enabled = HAS_NUMBA

        # Centralized JIT lifecycle
        if self.jit_enabled:
            if hasattr(self, '_setup_jit_functions'):
                self._setup_jit_functions()
            if eager_compile:
                self.compile()

    def compile(self) -> None:
        """
        Triggers JIT compilation for all core handles by running dummy steps.
        This eliminates lazy-compilation delays during the first real simulation.
        """
        if not self.jit_enabled:
            return

        # 1. Ensure JIT handles are created
        if hasattr(self, '_setup_jit_functions') and not hasattr(self, '_ode_jit') and not hasattr(self, '_forward_jit'):
            self._setup_jit_functions()

        # 2. Warm up basic function handles if they exist
        dummy_x = np.ones(self.dim)
        for attr in ['_ode_jit', '_forward_jit', '_jac_jit']:
            if hasattr(self, attr):
                handle = getattr(self, attr)
                _ = handle(dummy_x)

        # 3. Trigger simulation loop compilation (subclass-specific)
        self._warmup_simulation()

    def _warmup_simulation(self) -> None:
        """Triggers a minimal 1-step simulation to warm up the integration/map loops."""
        dummy_x = np.ones(self.dim)
        
        # ODE Path
        if hasattr(self, '_ode_jit'):
            # Pre-compile ALL standard RK methods (RK1-RK4)
            for m in ['RK1', 'RK2', 'RK3', 'RK4']:
                self.simulate(dt=0.01, t_span=(0, 0.01), y0=dummy_x, method=m)
                if hasattr(self, '_jac_jit'):
                    self.simulate_var(dt=0.01, t_span=(0, 0.01), x0=dummy_x, Phi0=np.eye(self.dim), method=m)
        
        # Map Path
        elif hasattr(self, '_forward_jit'):
            # simulate(x0, n_steps)
            self.simulate(x0=dummy_x, n_steps=1)