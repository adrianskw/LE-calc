import numpy as np

class DiscreteMap:
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

    def forward_map(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jac(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def simulate(self, x0, n_steps):
        self.n_steps = n_steps
        self.x: np.ndarray = np.empty((self.n_steps, self.dim))

        x0 = np.atleast_1d(np.asarray(x0, dtype=float))   # handle scalars & arrays
        if len(x0) != self.dim:
            raise ValueError(f"{type(self).__name__} expects a {self.dim}-D initial condition, but got x0 with length {len(x0)}.")
        self.x = np.zeros((self.n_steps, self.dim))
        for _ in range(min(1000, self.n_steps // 2)):            # warmup: discard transient
            x0 = self.forward_map(x0)
        for i in range(self.n_steps):
            self.x[i] = x0
            x0 = self.forward_map(x0)

    def discrete_qr_lyapunov_spectrum(self):
        # discrete QR, i.e. renorm at every step
        logAbsDiagR = np.zeros((self.n_steps, self.dim))
        Q = np.eye(self.dim)
        for i in range(self.n_steps):
            Q,R = np.linalg.qr(self.jac(self.x[i])@Q)
            logAbsDiagR[i] = np.log(np.abs(np.diag(R)))
        self.lyapunov_spectrum = np.mean(logAbsDiagR, axis=0)
        print(f"Lyapunov Spectrum: {np.array2string(self.lyapunov_spectrum, formatter={'float_kind':lambda x: f'{x:.5f}'})}")

    

class LogisticMap(DiscreteMap):
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
        super().__init__(dim=1)
        self.r = r

    def forward_map(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the next mapped value x_{n+1}.
        """
        return self.r * x * (1.0 - x)

    def jac(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the analytical Jacobian matrix (1x1).
        """
        return np.array([[self.r * (1.0 - 2.0 * x.item())]])


class HenonMap(DiscreteMap):
    """
    The Hénon map. A 2D discrete-time dynamical system that exhibits chaotic behavior.
    
    Parameters
    ----------
    a : float (default: 1.4)
    b : float (default: 0.3)
    """

    def __init__(self, a: float = 1.4, b: float = 0.3):
        super().__init__(dim=2)
        self.a = a
        self.b = b

    def forward_map(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the next mapped state.
        """
        x1, x2 = x
        return np.array([
            1.0 - self.a * x1**2 + x2,
            self.b * x1
        ])

    def jac(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the analytical Jacobian matrix.
        """
        x1, _ = x
        return np.array([
            [-2.0 * self.a * x1, 1.0],
            [self.b,             0.0]
        ])
