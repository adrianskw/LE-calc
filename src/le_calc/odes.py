"""
odes.py — Continuous-time dynamical systems (ODEs).

Each system defines:
  - ode(x)  : the vector field  f(x) = dx/dt
  - jac(x)  : the analytical Jacobian  J(x) = df/dx  at a single 1-D state

When JIT is available (self.jit_enabled is True), @njit-compiled closures are
built at construction time and used for the tight integration loops, giving a
significant speedup. When JIT is disabled, identical plain-Python functions are
used via the RK_METHODS / RK_VAR_METHODS lookup tables in utils.py.
"""

import numpy as np
from .base import DynamicalSystem
from .utils import njit, qr_2x2, qr_3x3, RK_METHODS, RK_VAR_METHODS


class ODEs(DynamicalSystem):
    """
    Base class for continuous-time dynamical systems (ODEs).
    Provides methods to define the vector field and Jacobian.
    """

    def __init__(self, dim: int, eager_compile: bool = True):
        super().__init__(dim=dim, eager_compile=eager_compile)

    def ode(self, x: np.ndarray) -> np.ndarray:
        """Compute the vector field f(x) = dx/dt."""
        raise NotImplementedError

    def jac(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the analytical Jacobian J(x) = df/dx at a single state.

        Parameters
        ----------
        x : np.ndarray, shape (dim,)

        Returns
        -------
        J : np.ndarray, shape (dim, dim)
        """
        raise NotImplementedError

    def calc_jac(self) -> np.ndarray:
        """
        Compute and store the Jacobian matrices along the stored trajectory.

        Iterates over self.x, calling self.jac(state) for each 1D state and
        storing the result in self.J with shape (n_steps, dim, dim).

        Returns
        -------
        self.J : np.ndarray, shape (n_steps, dim, dim)
        """
        self.J = np.empty((self.n_steps, self.dim, self.dim))
        
        # Use JIT-compiled jac if available and JIT is enabled
        jac_func = getattr(self, '_jac_jit', self.jac) if self.jit_enabled else self.jac
        
        for i in range(self.n_steps):
            self.J[i] = jac_func(self.x[i])
        return self.J

    def _get_simulation_handles(self, method: str, is_var: bool = False):
        """
        Helper to select the correct stepper and function handles based on JIT status.
        """
        lookup = RK_VAR_METHODS if is_var else RK_METHODS
        if method not in lookup:
            raise ValueError(f"Method '{method}' is not supported. "
                             f"Choose from {list(lookup)}.")

        jit_step, py_step = lookup[method]
        
        # Determine if we can use JIT handles
        use_jit = self.jit_enabled and hasattr(self, '_ode_jit')
        if is_var:
            use_jit = use_jit and hasattr(self, '_jac_jit')

        if use_jit:
            return (jit_step, self._ode_jit, self._jac_jit) if is_var else (jit_step, self._ode_jit)
        else:
            return (py_step, self.ode, self.jac) if is_var else (py_step, self.ode)

    def simulate(
        self,
        dt: float,
        t_span: tuple[float, float],
        y0: np.ndarray,
        method: str = 'RK4',
    ) -> np.ndarray:
        """Integrate the ODE system (state only) and store the trajectory."""
        step_func, ode_func = self._get_simulation_handles(method, is_var=False)

        t_eval = np.arange(t_span[0], t_span[1], dt)
        self.n_steps = len(t_eval)
        y = np.asarray(y0, dtype=float)
        self.x = np.zeros((self.n_steps, self.dim))

        # 1. Burn transients
        for _ in np.arange(0, t_span[0], dt):
            y = step_func(ode_func, dt, y)

        # 2. Integration loop
        for i in range(self.n_steps):
            self.x[i] = y
            y = step_func(ode_func, dt, y)

        return self.x

    def simulate_var(
        self,
        dt: float,
        t_span: tuple[float, float],
        x0: np.ndarray,
        Phi0: np.ndarray,
        method: str = 'RK4',
        qr_method: str = 'householder',
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate state + variational equations with QR re-orthonormalization.
        Stores the state trajectory in self.x.
        """
        step_func, ode_func, jac_func = self._get_simulation_handles(method, is_var=True)

        if qr_method == 'gram-schmidt' and self.dim == 2:
            qr_func = qr_2x2
        elif qr_method == 'gram-schmidt' and self.dim == 3:
            qr_func = qr_3x3
        else:
            qr_func = np.linalg.qr

        t_eval = np.arange(t_span[0], t_span[1], dt)
        self.n_steps = len(t_eval)
        y   = np.asarray(x0,   dtype=float)
        Phi = np.asarray(Phi0, dtype=float)
        self.x   = np.zeros((self.n_steps, self.dim))
        self.phi = np.zeros((self.n_steps, self.dim, self.dim))
        self.Q   = np.zeros((self.n_steps, self.dim, self.dim))
        self.R   = np.zeros((self.n_steps, self.dim, self.dim))

        # 1. Burn transients
        for _ in np.arange(0, t_span[0], dt):
            Q, _ = qr_func(Phi)
            y, Phi = step_func(ode_func, jac_func, dt, y, Q)

        # 2. Main integration loop
        for i in range(self.n_steps):
            self.x[i] = y
            Q, R = qr_func(Phi)
            self.phi[i], self.Q[i], self.R[i] = Phi, Q, R
            y, Phi = step_func(ode_func, jac_func, dt, y, Q)

        return self.x, self.phi, self.Q, self.R


# ---------------------------------------------------------------------------
# Concrete systems
# ---------------------------------------------------------------------------

class Lorenz63(ODEs):
    """
    The classic Lorenz 1963 system.

    Parameters
    ----------
    sigma : float  — Prandtl number (default 10.0)
    rho   : float  — Rayleigh number (default 28.0)
    beta  : float  — Geometric factor (default 8/3)
    """

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, 
                 beta: float = 8.0 / 3.0, eager_compile: bool = True):
        self.sigma = sigma
        self.rho   = rho
        self.beta  = beta
        super().__init__(dim=3, eager_compile=eager_compile)

    def _setup_jit_functions(self):
        """Build @njit-compiled closures with baked-in parameters."""
        s, r, b = self.sigma, self.rho, self.beta

        @njit
        def ode_jit(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([s*(x2-x1), x1*(r-x3)-x2, x1*x2-b*x3])

        @njit
        def jac_jit(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([[-s,    s,   0.0],
                              [r-x3, -1.0, -x1],
                              [x2,   x1,  -b ]])

        self._ode_jit = ode_jit
        self._jac_jit = jac_jit

    def ode(self, x: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x
        return np.array([
            self.sigma * (x2 - x1),
            x1 * (self.rho - x3) - x2,
            x1 * x2 - self.beta * x3,
        ])

    def jac(self, x: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x
        return np.array([
            [-self.sigma,  self.sigma, 0.0       ],
            [self.rho-x3, -1.0,       -x1       ],
            [x2,           x1,        -self.beta ],
        ])


class Rossler(ODEs):
    """
    The Rössler chaotic attractor.

    Parameters
    ----------
    a : float (default 0.2)
    b : float (default 0.2)
    c : float (default 5.7)
    """

    def __init__(self, a: float = 0.2, b: float = 0.2, 
                 c: float = 5.7, eager_compile: bool = True):
        self.a = a
        self.b = b
        self.c = c
        super().__init__(dim=3, eager_compile=eager_compile)

    def _setup_jit_functions(self):
        """Build @njit-compiled closures with baked-in parameters."""
        a, b, c = self.a, self.b, self.c

        @njit
        def ode_jit(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([-x2-x3, x1+a*x2, b+x3*(x1-c)])

        @njit
        def jac_jit(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return np.array([[0.0, -1.0, -1.0],
                             [1.0,  a,    0.0],
                             [x3,   0.0, x1-c]])

        self._ode_jit = ode_jit
        self._jac_jit = jac_jit

    def ode(self, x: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x
        return np.array([
            -x2 - x3,
            x1 + self.a * x2,
            self.b + x3 * (x1 - self.c),
        ])

    def jac(self, x: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x
        return np.array([
            [0.0, -1.0,    -1.0        ],
            [1.0,  self.a,  0.0        ],
            [x3,   0.0,    x1-self.c  ],
        ])