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
    HAS_NUMBA = False
    def njit(func=None, **kwargs):
        """No-op decorator: returns the function unchanged when Numba is absent."""
        if func is not None:
            return func          # @njit without arguments
        return lambda f: f       # @njit(...) with arguments


# ---------------------------------------------------------------------------
# Analytical QR decompositions (Gram-Schmidt, small fixed dimensions)
# ---------------------------------------------------------------------------

@njit(cache=True)
def qr_GS_2x2(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytical 2x2 QR decomposition (Modified Gram-Schmidt)."""
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
    """Analytical 3x3 QR decomposition (Modified Gram-Schmidt)."""
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
    """JIT-compiled wrapper for np.linalg.qr."""
    return np.linalg.qr(A)
    

# ---------------------------------------------------------------------------
# Runge-Kutta steppers — JIT-compiled versions
# All accept (ode_func, dt, y) or (ode_func, jac_func, dt, y, Phi).
# ---------------------------------------------------------------------------


@njit
def rk2(ode_func, dt: float, y: np.ndarray) -> np.ndarray:
    """Midpoint (RK2) step."""
    k1 = ode_func(y)
    k2 = ode_func(y + 0.5 * dt * k1)
    return y + dt * k2


@njit
def rk2_var(ode_func, jac_func, dt: float,
            y: np.ndarray, Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint (RK2) step for state + variational equation."""
    k1 = ode_func(y)
    L1 = jac_func(y) @ Phi
    k2 = ode_func(y + 0.5 * dt * k1)
    L2 = jac_func(y + 0.5 * dt * k1) @ (Phi + 0.5 * dt * L1)
    return y + dt * k2, Phi + dt * L2


@njit
def rk4(ode_func, dt: float, y: np.ndarray) -> np.ndarray:
    """Classic Runge-Kutta 4th-order (RK4) step."""
    k1 = ode_func(y)
    k2 = ode_func(y + 0.5 * dt * k1)
    k3 = ode_func(y + 0.5 * dt * k2)
    k4 = ode_func(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit
def rk4_var(ode_func, jac_func, dt: float,
            y: np.ndarray, Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classic Runge-Kutta 4th-order (RK4) step for state + variational equation."""
    k1 = ode_func(y)
    L1 = jac_func(y) @ Phi
    k2 = ode_func(y + 0.5 * dt * k1)
    L2 = jac_func(y + 0.5 * dt * k1) @ (Phi + 0.5 * dt * L1)
    k3 = ode_func(y + 0.5 * dt * k2)
    L3 = jac_func(y + 0.5 * dt * k2) @ (Phi + 0.5 * dt * L2)
    k4 = ode_func(y + dt * k3)
    L4 = jac_func(y + dt * k3) @ (Phi + dt * L3)
    y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    Phi_next = Phi + (dt / 6.0) * (L1 + 2.0 * L2 + 2.0 * L3 + L4)
    return y_next, Phi_next

RK_METHODS = {
    'RK2': rk2,
    'RK4': rk4,
}

RK_VAR_METHODS = {
    'RK2': rk2_var,
    'RK4': rk4_var,
}

# ---------------------------------------------------------------------------
# Discrete map simulation utilities
# ---------------------------------------------------------------------------

@njit
def simulate_map(map_func, x0: np.ndarray, n_steps: int, n_burn: int, dim: int) -> np.ndarray:
    """Generic JIT-compiled simulation loop for discrete-time maps (unified loop)."""
    x_hist = np.empty((n_steps, dim))
    x_curr = x0

    for i in range(n_burn + n_steps):
        if i >= n_burn:
            x_hist[i - n_burn] = x_curr
        
        x_curr = map_func(x_curr)
        
    return x_hist


@njit
def simulate_ode(step_func, ode_func, dt, n_steps, n_burn, x0, dim):
    """Generic JIT-compiled simulation loop for ODE state only."""
    x_hist = np.empty((n_steps, dim))

    for i in range(n_burn + n_steps):
        if i >= n_burn:
            x_hist[i - n_burn] = x0
        
        x0 = step_func(ode_func, dt, x0)
        
    return x_hist


@njit
def simulate_ode_var(step_func, ode_func, jac_func, qr_func, dt, n_steps, n_burn, x0, Phi0, dim):
    """Generic JIT-compiled simulation loop for ODE state + variational equation."""
    # Pre-allocate histories
    x_hist = np.empty((n_steps, dim))
    Phi_hist = np.empty((n_steps, dim, dim))
    Q_hist = np.empty((n_steps, dim, dim))
    R_hist = np.empty((n_steps, dim, dim))
    
    # Workspaces
    Q_work = np.empty((dim, dim))

    for i in range(n_burn + n_steps):
        # 1. Perform QR re-orthonormalization
        if i < n_burn:
            Q_work[:], _ = qr_func(Phi0)
            Q_basis = Q_work
        else:
            idx = i - n_burn
            x_hist[idx] = x0
            Phi_hist[idx] = Phi0
            
            q_out, r_out = qr_func(Phi0)
            Q_hist[idx], R_hist[idx] = q_out, r_out
            Q_basis = Q_hist[idx]
        
        # 2. Advance one step using the orthogonalized basis
        x0, Phi0 = step_func(ode_func, jac_func, dt, x0, Q_basis)
        
    return x_hist, Phi_hist, Q_hist, R_hist