"""
utils.py — Shared, performance-critical utilities.

JIT compilation is handled via Numba's @njit. A no-op decorator fallback
keeps the module importable without Numba. Use HAS_NUMBA to branch on
actual availability.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    njit = lambda f=None, **k: f if callable(f) else (lambda g: g)


# ---------------------------------------------------------------------------
# QR decompositions via Modified Gram-Schmidt (fixed small dimensions)
# ---------------------------------------------------------------------------

@njit(cache=True)
def qr_GS_2x2(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    2×2 QR decomposition using Modified Gram-Schmidt.

    Modified Gram-Schmidt (MGS) orthogonalizes columns sequentially,
    subtracting projections onto already-computed basis vectors before
    normalizing. MGS is numerically superior to classical Gram-Schmidt
    and avoids LAPACK overhead for these tiny fixed dimensions.

    Parameters
    ----------
    A : np.ndarray, shape (2, 2)

    Returns
    -------
    Q : np.ndarray, shape (2, 2)  — orthonormal columns
    R : np.ndarray, shape (2, 2)  — upper triangular
    """
    a00, a10 = A[0, 0], A[1, 0]
    r11 = np.sqrt(a00*a00 + a10*a10)
    q00, q10 = a00 / r11, a10 / r11

    a01, a11 = A[0, 1], A[1, 1]
    r12 = q00*a01 + q10*a11

    # Second basis vector is the 90° rotation of the first (exact in 2-D)
    q01, q11 = -q10, q00
    r22 = q01*a01 + q11*a11

    Q = np.empty((2, 2))
    Q[0, 0], Q[1, 0] = q00, q10
    Q[0, 1], Q[1, 1] = q01, q11

    R = np.zeros((2, 2))
    R[0, 0], R[0, 1], R[1, 1] = r11, r12, r22
    return Q, R


@njit(cache=True)
def qr_GS_3x3(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    3×3 QR decomposition using Modified Gram-Schmidt.

    See qr_GS_2x2 for an explanation of the MGS algorithm.

    Parameters
    ----------
    A : np.ndarray, shape (3, 3)

    Returns
    -------
    Q : np.ndarray, shape (3, 3)  — orthonormal columns
    R : np.ndarray, shape (3, 3)  — upper triangular
    """
    a00, a10, a20 = A[0, 0], A[1, 0], A[2, 0]
    a01, a11, a21 = A[0, 1], A[1, 1], A[2, 1]
    a02, a12, a22 = A[0, 2], A[1, 2], A[2, 2]

    r11 = np.sqrt(a00*a00 + a10*a10 + a20*a20)
    q00, q10, q20 = a00/r11, a10/r11, a20/r11

    r12 = q00*a01 + q10*a11 + q20*a21
    v01, v11, v21 = a01 - r12*q00, a11 - r12*q10, a21 - r12*q20
    r22 = np.sqrt(v01*v01 + v11*v11 + v21*v21)
    q01, q11, q21 = v01/r22, v11/r22, v21/r22

    r13 = q00*a02 + q10*a12 + q20*a22
    r23 = q01*a02 + q11*a12 + q21*a22
    v02 = a02 - r13*q00 - r23*q01
    v12 = a12 - r13*q10 - r23*q11
    v22 = a22 - r13*q20 - r23*q21
    r33 = np.sqrt(v02*v02 + v12*v12 + v22*v22)
    q02, q12, q22 = v02/r33, v12/r33, v22/r33

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
    QR decomposition via Householder reflections (delegates to np.linalg.qr).

    Householder is numerically more stable than Gram-Schmidt and preferred
    for dimensions ≥ 4 or when maximum accuracy is required. LAPACK overhead
    makes it slower than the hand-unrolled MGS variants for 2×2 and 3×3.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)

    Returns
    -------
    Q : np.ndarray  — orthonormal columns
    R : np.ndarray  — upper triangular
    """
    return np.linalg.qr(A)


# ---------------------------------------------------------------------------
# Runge-Kutta steppers
# ---------------------------------------------------------------------------

@njit(cache=True)
def rk2(ode_func: callable, dt: float, y: np.ndarray) -> np.ndarray:
    """
    Single step — Midpoint (RK2) method.

    Parameters
    ----------
    ode_func : callable  — vector field f(x)
    dt       : float     — time step
    y        : np.ndarray — current state

    Returns
    -------
    y_next : np.ndarray
    """
    k1 = ode_func(y)
    k2 = ode_func(y + 0.5*dt*k1)
    return y + dt*k2


@njit(cache=True)
def rk2_var(ode_func: callable, jac_func: callable, dt: float,
            y: np.ndarray, Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single step for state + variational equations — Midpoint (RK2).

    Integrates the coupled system:
        ẏ   = f(y)
        Φ̇   = J(y) Φ

    Parameters
    ----------
    ode_func : callable  — vector field f(x)
    jac_func : callable  — Jacobian J(x) = df/dx
    dt       : float     — time step
    y        : np.ndarray — current state
    Phi      : np.ndarray — current fundamental matrix / basis frame

    Returns
    -------
    y_next, Phi_next, J_curr : tuple
        J_curr is the Jacobian evaluated at the start of the step,
        used for continuous-QR spectrum computation.
    """
    k1     = ode_func(y)
    J_curr = jac_func(y)
    L1     = J_curr @ Phi
    k2     = ode_func(y + 0.5*dt*k1)
    L2     = jac_func(y + 0.5*dt*k1) @ (Phi + 0.5*dt*L1)
    return y + dt*k2, Phi + dt*L2, J_curr


@njit(cache=True)
def rk4(ode_func: callable, dt: float, y: np.ndarray) -> np.ndarray:
    """
    Single step — Classic RK4 method.

    Parameters
    ----------
    ode_func : callable  — vector field f(x)
    dt       : float     — time step
    y        : np.ndarray — current state

    Returns
    -------
    y_next : np.ndarray
    """
    k1 = ode_func(y)
    k2 = ode_func(y + 0.5*dt*k1)
    k3 = ode_func(y + 0.5*dt*k2)
    k4 = ode_func(y + dt*k3)
    return y + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


@njit(cache=True)
def rk4_var(ode_func: callable, jac_func: callable, dt: float,
            y: np.ndarray, Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single step for state + variational equations — Classic RK4.

    See rk2_var for the equations being integrated.

    Parameters
    ----------
    ode_func : callable
    jac_func : callable
    dt       : float
    y        : np.ndarray
    Phi      : np.ndarray

    Returns
    -------
    y_next, Phi_next, J_curr : tuple
    """
    k1     = ode_func(y)
    J_curr = jac_func(y)
    L1     = J_curr @ Phi
    k2     = ode_func(y + 0.5*dt*k1)
    L2     = jac_func(y + 0.5*dt*k1) @ (Phi + 0.5*dt*L1)
    k3     = ode_func(y + 0.5*dt*k2)
    L3     = jac_func(y + 0.5*dt*k2) @ (Phi + 0.5*dt*L2)
    k4     = ode_func(y + dt*k3)
    L4     = jac_func(y + dt*k3) @ (Phi + dt*L3)
    y_next   = y   + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    Phi_next = Phi + (dt/6.0) * (L1 + 2.0*L2 + 2.0*L3 + L4)
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
    'householder'     : qr_HH,
}


# ---------------------------------------------------------------------------
# Simulation kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def simulate_map(map_func: callable, x0: np.ndarray,
                 n_steps: int, n_burn: int, dim: int) -> np.ndarray:
    """
    Simulation loop for discrete-time maps.

    Parameters
    ----------
    map_func : callable  — map equation x_{n+1} = f(x_n)
    x0       : np.ndarray — initial condition
    n_steps  : int        — number of steps to record
    n_burn   : int        — transient steps to discard before recording
    dim      : int        — state dimension

    Returns
    -------
    x_hist : np.ndarray, shape (n_steps, dim)
    """
    x_hist = np.empty((n_steps, dim))
    x = x0
    for i in range(n_burn + n_steps):
        if i >= n_burn:
            x_hist[i - n_burn] = x
        x = map_func(x)
    return x_hist


@njit(cache=True)
def simulate_ode(step_func: callable, ode_func: callable, dt: float,
                 n_steps: int, n_burn: int,
                 x0: np.ndarray, dim: int) -> np.ndarray:
    """
    Simulation loop for ODE state integration (no variational equations).

    Parameters
    ----------
    step_func : callable  — RK stepper, e.g. rk4
    ode_func  : callable  — vector field f(x)
    dt        : float
    n_steps   : int       — steps to record
    n_burn    : int       — burn-in steps to discard
    x0        : np.ndarray — initial condition
    dim       : int

    Returns
    -------
    x_hist : np.ndarray, shape (n_steps, dim)
    """
    x_hist = np.empty((n_steps, dim))
    x = x0
    for i in range(n_burn + n_steps):
        if i >= n_burn:
            x_hist[i - n_burn] = x
        x = step_func(ode_func, dt, x)
    return x_hist


@njit(cache=True)
def simulate_ode_var(step_func: callable, ode_func: callable, jac_func: callable,
                     qr_func: callable, dt: float, n_steps: int, n_burn: int,
                     x0: np.ndarray, Phi0: np.ndarray,
                     dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulation loop integrating state + variational equations with QR re-orthonormalization.

    At each step the current basis frame is QR-factored before being advanced.
    The diagonal of R encodes local stretching rates; accumulating log|R_ii|
    over time gives the Lyapunov spectrum via the discrete-QR formula.

    Parameters
    ----------
    step_func : callable  — variational RK stepper (rk2_var or rk4_var)
    ode_func  : callable  — vector field f(x)
    jac_func  : callable  — Jacobian J(x) = df/dx
    qr_func   : callable  — QR decomposition (qr_GS_* or qr_HH)
    dt        : float
    n_steps   : int       — steps to record
    n_burn    : int       — burn-in steps to discard
    x0        : np.ndarray, shape (dim,)
    Phi0      : np.ndarray, shape (dim, dim) — initial basis frame (usually I)
    dim       : int

    Returns
    -------
    x_hist   : np.ndarray, shape (n_steps, dim)
    Phi_hist : np.ndarray, shape (n_steps, dim, dim) — pre-QR fundamental matrices
    Q_hist   : np.ndarray, shape (n_steps, dim, dim) — orthonormal frames
    R_hist   : np.ndarray, shape (n_steps, dim, dim) — upper-triangular growth factors
    J_hist   : np.ndarray, shape (n_steps, dim, dim) — Jacobians along the trajectory
    """
    x_hist   = np.empty((n_steps, dim))
    Phi_hist = np.empty((n_steps, dim, dim))
    Q_hist   = np.empty((n_steps, dim, dim))
    R_hist   = np.empty((n_steps, dim, dim))
    J_hist   = np.empty((n_steps, dim, dim))

    Q_curr = Phi0
    for i in range(n_burn + n_steps):
        Q, R = qr_func(Q_curr)

        if i >= n_burn:
            idx = i - n_burn
            x_hist[idx]           = x0
            Phi_hist[idx]         = Q_curr
            Q_hist[idx], R_hist[idx] = Q, R

        x0, Q_curr, J_curr = step_func(ode_func, jac_func, dt, x0,
                                        np.ascontiguousarray(Q))
        if i >= n_burn:
            J_hist[i - n_burn] = J_curr

    return x_hist, Phi_hist, Q_hist, R_hist, J_hist