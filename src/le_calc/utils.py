"""
utils.py — Shared utilities that are often called and should be optimized.

JIT compilation is handled via Numba's @njit. A no-op fallback is provided
so the module remains importable without Numba installed. The HAS_NUMBA flag
allows callers to branch on actual availability.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA, njit = False, lambda f=None, **k: f if callable(f) else (lambda g: g)


# ---------------------------------------------------------------------------
# Analytical QR decompositions (Gram-Schmidt, small fixed dimensions)
# ---------------------------------------------------------------------------

@njit(cache=True)
def qr_GS_2x2(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytical 2x2 QR decomposition using Modified Gram-Schmidt.

    Parameters
    ----------
    A : np.ndarray
        The 2x2 matrix to decompose.

    Returns
    -------
    Q : np.ndarray
        Orthogonal matrix (2x2).
    R : np.ndarray
        Upper triangular matrix (2x2).
    """
    a00, a10 = A[0, 0], A[1, 0]
    r11 = np.sqrt(a00*a00 + a10*a10)
    q00, q10 = a00 / r11, a10 / r11

    a01, a11 = A[0, 1], A[1, 1]
    r12 = q00 * a01 + q10 * a11

    q01, q11 = -q10, q00
    r22 = q01 * a01 + q11 * a11

    Q = np.empty((2, 2))
    Q[0, 0], Q[1, 0] = q00, q10
    Q[0, 1], Q[1, 1] = q01, q11

    R = np.zeros((2, 2))
    R[0, 0], R[0, 1], R[1, 1] = r11, r12, r22

    return Q, R


@njit(cache=True)
def qr_GS_3x3(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytical 3x3 QR decomposition using Modified Gram-Schmidt.

    Parameters
    ----------
    A : np.ndarray
        The 3x3 matrix to decompose.

    Returns
    -------
    Q : np.ndarray
        Orthogonal matrix (3x3).
    R : np.ndarray
        Upper triangular matrix (3x3).
    """
    a00, a10, a20 = A[0, 0], A[1, 0], A[2, 0]
    a01, a11, a21 = A[0, 1], A[1, 1], A[2, 1]
    a02, a12, a22 = A[0, 2], A[1, 2], A[2, 2]

    # Column 1
    r11 = np.sqrt(a00*a00 + a10*a10 + a20*a20)
    q00, q10, q20 = a00 / r11, a10 / r11, a20 / r11

    # Column 2
    r12 = q00*a01 + q10*a11 + q20*a21
    v01, v11, v21 = a01 - r12*q00, a11 - r12*q10, a21 - r12*q20
    r22 = np.sqrt(v01*v01 + v11*v11 + v21*v21)
    q01, q11, q21 = v01 / r22, v11 / r22, v21 / r22

    # Column 3
    r13 = q00*a02 + q10*a12 + q20*a22
    r23 = q01*a02 + q11*a12 + q21*a22
    v02, v12, v22 = (a02 - r13*q00 - r23*q01,
                     a12 - r13*q10 - r23*q11,
                     a22 - r13*q20 - r23*q21)
    r33 = np.sqrt(v02*v02 + v12*v12 + v22*v22)
    q02, q12, q22 = v02 / r33, v12 / r33, v22 / r33

    Q = np.empty((3, 3))
    Q[0, 0], Q[1, 0], Q[2, 0] = q00, q10, q20
    Q[0, 1], Q[1, 1], Q[2, 1] = q01, q11, q21
    Q[0, 2], Q[1, 2], Q[2, 2] = q02, q12, q22

    R = np.zeros((3, 3))
    R[0, 0], R[0, 1], R[0, 2] = r11, r12, r13
    R[1, 1], R[1, 2] = r22, r23
    R[2, 2] = r33

    return Q, R

@njit(cache=True)
def qr_HH(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled wrapper for np.linalg.qr (Householder).

    Parameters
    ----------
    A : np.ndarray
        The matrix to decompose.

    Returns
    -------
    Q, R : tuple[np.ndarray, np.ndarray]
    """
    return np.linalg.qr(A)
    

# ---------------------------------------------------------------------------
# Runge-Kutta steppers — JIT-compiled versions
# ---------------------------------------------------------------------------


@njit(cache=True)
def rk2(ode_func: callable, dt: float, y: np.ndarray) -> np.ndarray:
    """
    Advance one step using the Midpoint (RK2) method.

    Parameters
    ----------
    ode_func : callable
        System vector field f(x).
    dt : float
        Time step.
    y : np.ndarray
        Current state.

    Returns
    -------
    y_next : np.ndarray
    """
    k1 = ode_func(y)
    k2 = ode_func(y + 0.5 * dt * k1)
    return y + dt * k2


@njit(cache=True)
def rk2_var(ode_func: callable, jac_func: callable, dt: float,
            y: np.ndarray, Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Advance one step for state and variational equations using RK2.

    Parameters
    ----------
    ode_func : callable
        System vector field f(x).
    jac_func : callable
        System Jacobian df/dx.
    dt : float
        Time step.
    y : np.ndarray
        Current state.
    Phi : np.ndarray
        Current fundamental matrix or basis frame.

    Returns
    -------
    y_next, Phi_next, J_curr : tuple
    """
    k1 = ode_func(y)
    J_curr = jac_func(y)
    L1 = J_curr @ Phi
    k2 = ode_func(y + 0.5 * dt * k1)
    L2 = jac_func(y + 0.5 * dt * k1) @ (Phi + 0.5 * dt * L1)
    return y + dt * k2, Phi + dt * L2, J_curr


@njit(cache=True)
def rk4(ode_func: callable, dt: float, y: np.ndarray) -> np.ndarray:
    """
    Advance one step using the classic RK4 method.

    Parameters
    ----------
    ode_func : callable
        System vector field.
    dt : float
        Time step.
    y : np.ndarray
        Current state.

    Returns
    -------
    y_next : np.ndarray
    """
    k1 = ode_func(y)
    k2 = ode_func(y + 0.5 * dt * k1)
    k3 = ode_func(y + 0.5 * dt * k2)
    k4 = ode_func(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=True)
def rk4_var(ode_func: callable, jac_func: callable, dt: float,
            y: np.ndarray, Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Advance one step for state and variational equations using RK4.

    Parameters
    ----------
    ode_func : callable
    jac_func : callable
    dt : float
    y : np.ndarray
    Phi : np.ndarray

    Returns
    -------
    y_next, Phi_next, J_curr : tuple
    """
    k1 = ode_func(y)
    J_curr = jac_func(y)
    L1 = J_curr @ Phi
    k2 = ode_func(y + 0.5 * dt * k1)
    L2 = jac_func(y + 0.5 * dt * k1) @ (Phi + 0.5 * dt * L1)
    k3 = ode_func(y + 0.5 * dt * k2)
    L3 = jac_func(y + 0.5 * dt * k2) @ (Phi + 0.5 * dt * L2)
    k4 = ode_func(y + dt * k3)
    L4 = jac_func(y + dt * k3) @ (Phi + dt * L3)
    y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    Phi_next = Phi + (dt / 6.0) * (L1 + 2.0 * L2 + 2.0 * L3 + L4)
    return y_next, Phi_next, J_curr

RK_METHODS = {
    'RK2': rk2,
    'RK4': rk4,
}

RK_VAR_METHODS = {
    'RK2': rk2_var,
    'RK4': rk4_var,
}

QR_METHODS = {
    'gram-schmidt-2x2': qr_GS_2x2,
    'gram-schmidt-3x3': qr_GS_3x3,
    'householder': qr_HH
}

# ---------------------------------------------------------------------------
# Trajectory simulation kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def simulate_map(map_func: callable, x0: np.ndarray, n_steps: int, n_burn: int, dim: int) -> np.ndarray:
    """
    Standard JIT-compiled simulation loop for discrete-time maps.

    Parameters
    ----------
    map_func : callable
        The map equation x_{n+1} = f(x_n).
    x0 : np.ndarray
        Initial condition.
    n_steps : int
        Number of steps to record.
    n_burn : int
        Number of transient steps to discard.
    dim : int
        System dimension.

    Returns
    -------
    x_hist : np.ndarray, shape (n_steps, dim)
    """
    x_hist = np.empty((n_steps, dim))
    x_curr = x0

    for i in range(n_burn + n_steps):
        if i >= n_burn:
            x_hist[i - n_burn] = x_curr
        
        x_curr = map_func(x_curr)
        
    return x_hist


@njit(cache=True)
def simulate_ode(step_func: callable, ode_func: callable, dt: float, 
                 n_steps: int, n_burn: int, x0: np.ndarray, dim: int) -> np.ndarray:
    """
    Standard JIT-compiled simulation loop for ODE state only.

    Parameters
    ----------
    step_func : callable
    ode_func : callable
    dt : float
    n_steps, n_burn, dim : int
    x0 : np.ndarray

    Returns
    -------
    x_hist : np.ndarray, shape (n_steps, dim)
    """
    x_hist = np.empty((n_steps, dim))

    for i in range(n_burn + n_steps):
        if i >= n_burn:
            x_hist[i - n_burn] = x0
        
        x0 = step_func(ode_func, dt, x0)
        
    return x_hist


@njit(cache=True)
def simulate_ode_var(step_func: callable, ode_func: callable, jac_func: callable, 
                     qr_func: callable, dt: float, n_steps: int, n_burn: int, 
                     x0: np.ndarray, Phi0: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performance kernel for the high-frequency integration + QR pipeline.

    Parameters
    ----------
    step_func : callable
        Variational RK stepper (rk2_var or rk4_var).
    ode_func : callable
    jac_func : callable
    qr_func : callable
    dt : float
    n_steps, n_burn, dim : int
    x0, Phi0 : np.ndarray

    Returns
    -------
    x_hist   : np.ndarray, shape (n_steps, dim)
    Phi_hist : np.ndarray, shape (n_steps, dim, dim)
    Q_hist   : np.ndarray, shape (n_steps, dim, dim)
    R_hist   : np.ndarray, shape (n_steps, dim, dim)
    J_hist   : np.ndarray, shape (n_steps, dim, dim)
    """
    # Pre-allocate histories
    x_hist   = np.empty((n_steps, dim))
    Phi_hist = np.empty((n_steps, dim, dim))
    Q_hist   = np.empty((n_steps, dim, dim))
    R_hist   = np.empty((n_steps, dim, dim))
    J_hist   = np.empty((n_steps, dim, dim))
    
    # Track the current basis frame (orthogonal fundamental matrix)
    Q_curr = Phi0

    for i in range(n_burn + n_steps):
        # 1. Perform QR re-orthonormalization
        q_out, r_out = qr_func(Q_curr)

        if i >= n_burn:
            idx = i - n_burn
            x_hist[idx] = x0
            Phi_hist[idx] = Q_curr
            Q_hist[idx], R_hist[idx] = q_out, r_out
        
        # 2. Advance one step using the orthogonalized basis (q_out)
        q_contig = np.ascontiguousarray(q_out)
        x0, Q_curr, J_curr = step_func(ode_func, jac_func, dt, x0, q_contig)

        if i >= n_burn:
            J_hist[i - n_burn] = J_curr
        
    return x_hist, Phi_hist, Q_hist, R_hist, J_hist