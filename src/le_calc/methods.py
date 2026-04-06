"""
methods.py — Lyapunov spectrum computation methods.

All public functions have both a NumPy fallback (for Numba-free environments)
and a @njit-compiled fast path. The JIT path is selected automatically when
HAS_NUMBA is True.
"""

import numpy as np
from .utils import njit, qr_GS_2x2, qr_GS_3x3, qr_HH, HAS_NUMBA


# ---------------------------------------------------------------------------
# Public spectrum methods
# ---------------------------------------------------------------------------

def local_lyapunov_exponents(Q_history: np.ndarray,
                              J_history: np.ndarray) -> np.ndarray:
    """
    Instantaneous (local) Lyapunov exponents from the continuous-QR formulation.

    Computes the diagonal of Q^T(t) J(t) Q(t) at each time step.
    The time-averaged diagonal gives the global Lyapunov spectrum.

    Parameters
    ----------
    Q_history : np.ndarray, shape (n_steps, dim, dim)  — orthonormal frames
    J_history : np.ndarray, shape (n_steps, dim, dim)  — Jacobians along the trajectory

    Returns
    -------
    chi : np.ndarray, shape (n_steps, dim)
        Instantaneous expansion rate for each Lyapunov direction.
    """
    if HAS_NUMBA:
        return _local_lyapunov_exponents_jit(Q_history, J_history)
    QTJQ = Q_history.transpose(0, 2, 1) @ J_history @ Q_history
    return np.diagonal(QTJQ, axis1=1, axis2=2)


def continuous_qr_spectrum(Q_history: np.ndarray,
                            J_history: np.ndarray) -> np.ndarray:
    """
    Lyapunov spectrum via the continuous-QR formulation.

    Averages the local Lyapunov exponents (diagonal of Q^T J Q) over time.

    Parameters
    ----------
    Q_history : np.ndarray, shape (n_steps, dim, dim)
    J_history : np.ndarray, shape (n_steps, dim, dim)

    Returns
    -------
    spectrum : np.ndarray, shape (dim,)
    """
    if HAS_NUMBA:
        return _continuous_qr_spectrum_jit(Q_history, J_history)
    return np.mean(local_lyapunov_exponents(Q_history, J_history), axis=0)


def discrete_qr_spectrum(R_history: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Lyapunov spectrum via the discrete-QR (Benettin) formulation.

    Uses the identity:
        λ_i = (1 / N·dt) Σ_n ln|R_ii(n)|

    where R comes from the QR re-orthonormalization at each step.

    Parameters
    ----------
    R_history : np.ndarray, shape (n_steps, dim, dim) — upper-triangular growth factors
    dt        : float — time step between re-orthonormalizations

    Returns
    -------
    spectrum : np.ndarray, shape (dim,)
    """
    if HAS_NUMBA:
        return _discrete_qr_spectrum_jit(R_history, dt)
    R_diag = np.diagonal(R_history, axis1=1, axis2=2)
    return np.mean(np.log(np.abs(R_diag)), axis=0) / dt


# ---------------------------------------------------------------------------
# Matrix-exponential / Taylor integration methods
# ---------------------------------------------------------------------------

@njit(cache=True)
def matrix_exponential_spectrum(J_history: np.ndarray, dt: float,
                                 qr_method: str = 'householder',
                                 order: int = 4) -> np.ndarray:
    """
    Lyapunov spectrum via direct matrix-exponential integration.

    Approximates exp(J·dt) with a truncated Taylor series and evolves an
    orthonormal basis frame through QR re-orthonormalization at each step.

    Taylor expansion:  exp(A) ≈ I + A + A²/2! + A³/3! + … + Aⁿ/n!

    Parameters
    ----------
    J_history : np.ndarray, shape (n_steps, dim, dim) — Jacobians along trajectory
    dt        : float
    qr_method : str   — 'householder' (default) or 'gram-schmidt'
    order     : int   — Taylor truncation order (higher → more accurate, slower)

    Returns
    -------
    spectrum : np.ndarray, shape (dim,)
    """
    n_steps, dim, _ = J_history.shape
    log_sums = np.zeros(dim)

    Q_curr = np.eye(dim)
    I_dim  = np.eye(dim)
    M      = np.empty((dim, dim))
    term   = np.empty((dim, dim))

    for i in range(n_steps):
        A = J_history[i] * dt

        # Build Taylor approximation of exp(A) into M.
        # M[:] and term[:] copy values in-place, preserving pre-allocated buffers.
        M[:]    = I_dim
        term[:] = I_dim
        for k in range(1, order + 1):
            term = (term @ A) / k
            M   += term

        if qr_method == 'gram-schmidt' and dim == 2:
            Q_next, R = qr_GS_2x2(M @ Q_curr)
        elif qr_method == 'gram-schmidt' and dim == 3:
            Q_next, R = qr_GS_3x3(M @ Q_curr)
        else:
            Q_next, R = qr_HH(M @ Q_curr)

        Q_curr = np.ascontiguousarray(Q_next)
        for d in range(dim):
            log_sums[d] += np.log(np.abs(R[d, d]))

    return log_sums / (n_steps * dt)


@njit(cache=True)
def taylor_spectrum(J_history: np.ndarray, xdot_H_history: np.ndarray,
                    dt: float, qr_method: str = 'householder') -> np.ndarray:
    """
    Lyapunov spectrum via a 4th-order Taylor series for exp(J·dt).

    Uses a precomputed Hessian contraction term (xdot_H) to include
    second-order flow information in the Taylor expansion:

        M = I + dt·J + (dt²/2)(J² + H) + (dt³/6)J³ + (dt⁴/24)J⁴

    where H = xdot_H is the contraction of the velocity field with the
    Hessian of the flow.

    Parameters
    ----------
    J_history      : np.ndarray, shape (n_steps, dim, dim)
    xdot_H_history : np.ndarray, shape (n_steps, dim, dim)
    dt             : float
    qr_method      : str — 'householder' (default) or 'gram-schmidt'

    Returns
    -------
    spectrum : np.ndarray, shape (dim,)
    """
    n_steps, dim, _ = J_history.shape
    log_sums = np.zeros(dim)

    Q_curr = np.eye(dim)
    I_dim  = np.eye(dim)

    for n in range(n_steps):
        J = J_history[n]
        H = xdot_H_history[n]
        J2 = J @ J
        M = (I_dim
             + dt    *  J
             + (dt**2/2)  * (J2 + H)
             + (dt**3/6)  * (J2 @ J)
             + (dt**4/24) * (J2 @ J2))

        if qr_method == 'gram-schmidt' and dim == 2:
            Q_next, R = qr_GS_2x2(M @ Q_curr)
        elif qr_method == 'gram-schmidt' and dim == 3:
            Q_next, R = qr_GS_3x3(M @ Q_curr)
        else:
            Q_next, R = qr_HH(M @ Q_curr)

        Q_curr = np.ascontiguousarray(Q_next)
        for d in range(dim):
            log_sums[d] += np.log(np.abs(R[d, d]))

    return log_sums / (n_steps * dt)


# ---------------------------------------------------------------------------
# Discrete-map QR loops
# ---------------------------------------------------------------------------

@njit(cache=True)
def discrete_qr_loop_2d(J: np.ndarray,
                         n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fused, allocation-free QR loop for 2-D discrete maps.

    Inlines the 2×2 MGS factorization directly to avoid function-call
    overhead inside the tight loop.

    Parameters
    ----------
    J       : np.ndarray, shape (n_steps, 2, 2) — Jacobians along orbit
    n_steps : int

    Returns
    -------
    Q_out : np.ndarray, shape (n_steps, 2, 2)
    R_out : np.ndarray, shape (n_steps, 2, 2)
    """
    Q     = np.eye(2)
    Q_out = np.zeros((n_steps, 2, 2))
    R_out = np.zeros((n_steps, 2, 2))

    for i in range(n_steps):
        # M = J[i] @ Q  (inlined for the 2×2 case)
        m00 = J[i, 0, 0]*Q[0, 0] + J[i, 0, 1]*Q[1, 0]
        m10 = J[i, 1, 0]*Q[0, 0] + J[i, 1, 1]*Q[1, 0]
        m01 = J[i, 0, 0]*Q[0, 1] + J[i, 0, 1]*Q[1, 1]
        m11 = J[i, 1, 0]*Q[0, 1] + J[i, 1, 1]*Q[1, 1]

        r11  = np.sqrt(m00*m00 + m10*m10)
        q00, q10 = m00/r11, m10/r11
        r12  = q00*m01 + q10*m11
        q01, q11 = -q10, q00
        r22  = q01*m01 + q11*m11

        Q[0, 0], Q[1, 0], Q[0, 1], Q[1, 1] = q00, q10, q01, q11
        Q_out[i] = Q
        R_out[i, 0, 0], R_out[i, 0, 1], R_out[i, 1, 1] = r11, r12, r22

    return Q_out, R_out


@njit(cache=True)
def discrete_qr_loop(qr_func: callable, J: np.ndarray,
                      n_steps: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generic QR loop for discrete maps (any dimension).

    Parameters
    ----------
    qr_func : callable               — QR decomposition (qr_GS_* or qr_HH)
    J       : np.ndarray, shape (n_steps, dim, dim) — Jacobians along orbit
    n_steps : int
    dim     : int

    Returns
    -------
    Q_out : np.ndarray, shape (n_steps, dim, dim)
    R_out : np.ndarray, shape (n_steps, dim, dim)
    """
    Q     = np.eye(dim)
    Q_out = np.zeros((n_steps, dim, dim))
    R_out = np.zeros((n_steps, dim, dim))

    for i in range(n_steps):
        Q, R      = qr_func(J[i] @ Q)
        Q_out[i]  = Q
        R_out[i]  = R

    return Q_out, R_out


# ---------------------------------------------------------------------------
# JIT-compiled kernels for public methods
# ---------------------------------------------------------------------------

@njit(cache=True)
def _discrete_qr_spectrum_jit(R_history: np.ndarray, dt: float) -> np.ndarray:
    """JIT kernel for discrete_qr_spectrum."""
    n_steps, dim, _ = R_history.shape
    log_sums = np.zeros(dim)
    for i in range(n_steps):
        for d in range(dim):
            log_sums[d] += np.log(np.abs(R_history[i, d, d]))
    return log_sums / (n_steps * dt)


@njit(cache=True)
def _local_lyapunov_exponents_jit(Q_history: np.ndarray,
                                   J_history: np.ndarray) -> np.ndarray:
    """JIT kernel for local_lyapunov_exponents."""
    n_steps, dim, _ = Q_history.shape
    res = np.empty((n_steps, dim))
    for i in range(n_steps):
        Q = Q_history[i]
        J = J_history[i]
        for d in range(dim):
            val = 0.0
            for j in range(dim):
                inner = 0.0
                for k in range(dim):
                    inner += J[j, k] * Q[k, d]
                val += Q[j, d] * inner
            res[i, d] = val
    return res


@njit(cache=True)
def _continuous_qr_spectrum_jit(Q_history: np.ndarray,
                                 J_history: np.ndarray) -> np.ndarray:
    """JIT kernel for continuous_qr_spectrum."""
    n_steps, dim, _ = Q_history.shape
    sums = np.zeros(dim)
    for i in range(n_steps):
        Q = Q_history[i]
        J = J_history[i]
        for d in range(dim):
            val = 0.0
            for j in range(dim):
                inner = 0.0
                for k in range(dim):
                    inner += J[j, k] * Q[k, d]
                val += Q[j, d] * inner
            sums[d] += val
    return sums / n_steps
