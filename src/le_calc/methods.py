"""
methods.py — methods for calculating Lyapunov exponents
"""

import numpy as np
from .utils import njit, qr_GS_2x2, qr_GS_3x3, qr_HH, HAS_NUMBA


def local_lyapunov_exponents(Q_history: np.ndarray, J_history: np.ndarray) -> np.ndarray:
    """
    Compute the local Lyapunov exponents from the continuous QR formulation.
    Formula: chi_i(t) = (Q^T(t) * J(t) * Q(t))_ii
    """
    if HAS_NUMBA:
        return local_lyapunov_exponents_jit(Q_history, J_history)

    # Batched matrix multiplication: Q^T @ J @ Q
    # Q_history.transpose(0, 2, 1) gives the transpose of each Q matrix in the stack
    QTJQ = Q_history.transpose(0, 2, 1) @ J_history @ Q_history
    # Extract the diagonal elements for each time step
    return np.diagonal(QTJQ, axis1=1, axis2=2)


def continuous_qr_spectrum(Q_history: np.ndarray, J_history: np.ndarray) -> np.ndarray:
    """Compute the Lyapunov spectrum using the continuous QR formulation with a simple mean."""
    if HAS_NUMBA:
        return continuous_qr_spectrum_jit(Q_history, J_history)

    local_lyap = local_lyapunov_exponents(Q_history, J_history)
    return np.mean(local_lyap, axis=0)


def discrete_qr_spectrum(R_history: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute the Lyapunov spectrum from the discrete QR formulation (history of R matrices).
    Formula: lambda_i = 1/(N*dt) * sum(ln|R_ii|)
    """
    # Extract the diagonal elements for each R matrix
    R_diag = np.diagonal(R_history, axis1=1, axis2=2)
    # Compute the mean log of the absolute diagonal elements
    return np.mean(np.log(np.abs(R_diag)), axis=0) / dt


@njit
def matrix_exponential_spectrum(
    J_history: np.ndarray, 
    dt: float, 
    qr_method: str = 'householder',
    order: int = 4
) -> np.ndarray:
    """
    Compute the Lyapunov spectrum using the Matrix Exponential formulation.
    Fully JIT-compiled and memory-optimized (O(1) storage).
    """
    n_steps, dim, _ = J_history.shape
    log_sums = np.zeros(dim)
    
    # 1. Pre-allocate integration workspaces
    Q_curr = np.eye(dim)
    I_dim = np.eye(dim)
    M = np.empty((dim, dim))
    term = np.empty((dim, dim))
    
    for i in range(n_steps):
        # 2. Taylor expansion for exp(J*dt) 
        A = J_history[i] * dt
        M[:] = I_dim
        term[:] = I_dim
        for k in range(1, order + 1):
            term = (term @ A) / k
            M += term
        
        # 3. Evolve basis frame
        if qr_method == 'gram-schmidt' and dim == 2:
            Q_next, R = qr_GS_2x2(M @ Q_curr)
        elif qr_method == 'gram-schmidt' and dim == 3:
            Q_next, R = qr_GS_3x3(M @ Q_curr)
        else:
            Q_next, R = qr_HH(M @ Q_curr)

        Q_curr = np.ascontiguousarray(Q_next)
        
        # 4. Accumulate growth directly (lambda_i = mean(ln|R_ii|) / dt)
        for d in range(dim):
            log_sums[d] += np.log(np.abs(R[d, d]))
            
    # 5. Final spectrum calculation
    return log_sums / (n_steps * dt)

@njit
def taylor_spectrum(
    J_history: np.ndarray, 
    xdot_H_history: np.ndarray,
    dt: float, 
    qr_method: str = 'householder'
) -> np.ndarray:
    """
    Compute the Lyapunov spectrum using the partial derivatives of the Taylor series expansion.
    Fully JIT-compiled and memory-optimized (O(1) storage).
    """
    n_steps, dim, _ = J_history.shape
    log_sums = np.zeros(dim)
    
    # 1. Pre-allocate integration workspaces
    Q_curr = np.eye(dim)
    I_dim = np.eye(dim)
    M = np.empty((dim, dim))
    
    for n in range(n_steps):
        # 2. Taylor expansion for exp(J*dt) 
        M = (I_dim + dt*J_history[n] + 
             (dt**2)/2 * (J_history[n] @ J_history[n] + xdot_H_history[n]) +
             (dt**3)/6 * (J_history[n] @ J_history[n] @ J_history[n]) +
             (dt**4)/24 * (J_history[n] @ J_history[n] @ J_history[n] @ J_history[n]))
        
        # 3. Evolve basis frame
        if qr_method == 'gram-schmidt' and dim == 2:
            Q_next, R = qr_GS_2x2(M @ Q_curr)
        elif qr_method == 'gram-schmidt' and dim == 3:
            Q_next, R = qr_GS_3x3(M @ Q_curr)
        else:
            Q_next, R = qr_HH(M @ Q_curr)

        Q_curr = np.ascontiguousarray(Q_next)
        
        # 4. Accumulate growth directly (lambda_i = mean(ln|R_ii|) / dt)
        for d in range(dim):
            log_sums[d] += np.log(np.abs(R[d, d]))
            
    # 5. Final spectrum calculation
    return log_sums / (n_steps * dt)

# ---------------------------------------------------------------------------
# JIT-Compiled Calculation Loops (Standardized)
# ---------------------------------------------------------------------------

@njit
def discrete_qr_loop_2d(J: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Highly optimized, fully-inlined 2x2 QR loop for discrete maps."""
    Q = np.eye(2)
    Q_out = np.zeros((n_steps, 2, 2))
    R_out = np.zeros((n_steps, 2, 2))
    for i in range(n_steps):
        # Analytical 2x2 QR
        m00 = J[i, 0, 0]*Q[0, 0] + J[i, 0, 1]*Q[1, 0]
        m10 = J[i, 1, 0]*Q[0, 0] + J[i, 1, 1]*Q[1, 0]
        m01 = J[i, 0, 0]*Q[0, 1] + J[i, 0, 1]*Q[1, 1]
        m11 = J[i, 1, 0]*Q[0, 1] + J[i, 1, 1]*Q[1, 1]
        r11 = np.sqrt(m00*m00 + m10*m10)
        q00, q10 = m00 / r11, m10 / r11
        r12 = q00 * m01 + q10 * m11
        q01, q11 = -q10, q00
        r22 = q01 * m01 + q11 * m11
        Q[0, 0], Q[1, 0], Q[0, 1], Q[1, 1] = q00, q10, q01, q11
        Q_out[i] = Q
        R_out[i, 0, 0], R_out[i, 0, 1], R_out[i, 1, 1] = r11, r12, r22
    return Q_out, R_out


@njit
def discrete_qr_loop(qr_func, J: np.ndarray, n_steps: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Generic JIT-compiled QR re-orthonormalization loop for discrete maps."""
    Q = np.eye(dim)
    Q_out = np.zeros((n_steps, dim, dim))
    R_out = np.zeros((n_steps, dim, dim))
    for i in range(n_steps):
        M = J[i] @ Q
        Q_new, R = qr_func(M)
        Q = Q_new
        Q_out[i] = Q
        R_out[i] = R
    return Q_out, R_out


@njit(cache=True)
def local_lyapunov_exponents_jit(Q_history: np.ndarray, J_history: np.ndarray) -> np.ndarray:
    """JIT-compiled local Lyapunov exponents calculation (O(1) intermediate memory)."""
    n_steps, dim, _ = Q_history.shape
    res = np.empty((n_steps, dim))
    for i in range(n_steps):
        Q = Q_history[i]
        J = J_history[i]
        for d in range(dim):
            # (Q^T * J * Q)_dd = sum_j sum_k Q_jd * J_jk * Q_kd
            val = 0.0
            for j in range(dim):
                row_sum = 0.0
                for k in range(dim):
                    row_sum += J[j, k] * Q[k, d]
                val += Q[j, d] * row_sum
            res[i, d] = val
    return res


@njit(cache=True)
def continuous_qr_spectrum_jit(Q_history: np.ndarray, J_history: np.ndarray) -> np.ndarray:
    """JIT-compiled continuous QR spectrum calculation (Mean of diagonal growth)."""
    n_steps, dim, _ = Q_history.shape
    sums = np.zeros(dim)
    for i in range(n_steps):
        Q = Q_history[i]
        J = J_history[i]
        for d in range(dim):
            val = 0.0
            for j in range(dim):
                row_sum = 0.0
                for k in range(dim):
                    row_sum += J[j, k] * Q[k, d]
                val += Q[j, d] * row_sum
            sums[d] += val
    return sums / n_steps
