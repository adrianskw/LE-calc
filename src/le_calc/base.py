"""
base.py — Abstract base class for all dynamical systems.
"""

import numpy as np
from .utils import HAS_NUMBA, QR_METHODS


class DynamicalSystem:
    """
    Base class for all dynamical systems (ODEs and discrete maps).

    Stores trajectory histories as attributes after calling simulate().
    Subclasses should override compile() to warm up their own JIT kernels,
    calling super().compile() to warm up the shared function handles.

    Attributes
    ----------
    dim               : int        — state-space dimension
    n_steps           : int        — number of recorded steps (set by simulate)
    jit_enabled       : bool       — True when Numba is available
    x                 : np.ndarray, shape (n_steps, dim)         — trajectory
    J                 : np.ndarray, shape (n_steps, dim, dim)    — Jacobians
    phi               : np.ndarray, shape (n_steps, dim, dim)    — fundamental matrices
    Q                 : np.ndarray, shape (n_steps, dim, dim)    — orthonormal frames
    R                 : np.ndarray, shape (n_steps, dim, dim)    — QR upper-triangular factors
    lyapunov_spectrum : np.ndarray, shape (dim,)
    xdot_H_history    : np.ndarray, shape (n_steps, dim, dim)    — Hessian contractions
    """

    def __init__(self, dim: int, **kwargs):
        """
        Parameters
        ----------
        dim            : int  — state-space dimension
        eager_compile  : bool — if True (default), warm up JIT on construction
        """
        eager_compile     = kwargs.get('eager_compile', True)
        self.dim          = dim
        self.n_steps      = 0
        self.jit_enabled  = HAS_NUMBA

        self.x                = np.empty((0, dim))
        self.lyapunov_spectrum = np.empty(dim)
        self.J = self.phi = self.Q = self.R = np.empty((0, dim, dim))
        self.xdot_H_history   = np.empty((0, dim, dim))

        if self.jit_enabled and eager_compile:
            self.compile()

    def _get_qr_func(self, qr_method: str) -> callable:
        """
        Return the QR callable for the requested method name.

        'gram-schmidt' resolves to the dimension-specific MGS variant;
        anything else falls back to Householder.
        """
        if qr_method == 'gram-schmidt':
            return QR_METHODS[f'gram-schmidt-{self.dim}x{self.dim}']
        return QR_METHODS['householder']

    def compile(self) -> None:
        """
        Warm up JIT compilation by running a single dummy call through each
        function handle (forward_map, ode, jac, xdot_H).

        Subclasses should override this, warm up their own kernels, then call
        super().compile().
        """
        if not self.jit_enabled:
            return
        ones = np.ones(self.dim)
        for attr in ['forward_map', 'ode', 'jac', 'xdot_H']:
            handle = getattr(self, attr, None)
            if callable(handle):
                try:
                    handle(ones, ones) if attr == 'xdot_H' else handle(ones)
                except Exception:
                    pass