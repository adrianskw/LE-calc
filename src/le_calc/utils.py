"""
utils.py — Shared utilities that are often called and should be optimized.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm


def rk4_step(model, dt: float, y: np.ndarray) -> np.ndarray:
    """Take a single Runge-Kutta 4th order step for the state only."""
    k1 = model.ode(y)
    k2 = model.ode(y + (dt / 2.0) * k1)
    k3 = model.ode(y + (dt / 2.0) * k2)
    k4 = model.ode(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step_variational(model, dt: float, y: np.ndarray, Phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Take a single Runge-Kutta 4th order step for both state and variational equation."""
    # State stages
    k1 = model.ode(y)
    k2 = model.ode(y + (dt / 2.0) * k1)
    k3 = model.ode(y + (dt / 2.0) * k2)
    k4 = model.ode(y + dt * k3)

    # Variational stages (dPhi/dt = J(x) @ Phi)
    L1 = model.jac(y)                   @  Phi
    L2 = model.jac(y + (dt / 2.0) * k1) @ (Phi + (dt / 2.0) * L1)
    L3 = model.jac(y + (dt / 2.0) * k2) @ (Phi + (dt / 2.0) * L2)
    L4 = model.jac(y +  dt        * k3) @ (Phi +  dt        * L3)

    y_next = y   + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    Phi_next = Phi + (dt / 6.0) * (L1 + 2.0 * L2 + 2.0 * L3 + L4)
    return y_next, Phi_next


"""Analytical 2x2 QR decomposition (Modified Gram-Schmidt)."""
# Optimized scalar math to avoid NumPy overhead for 2x2
def qr_2x2(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a00, a10 = A[0, 0], A[1, 0]
    r11 = np.sqrt(a00*a00 + a10*a10)
    q00, q10 = a00 / r11, a10 / r11
    
    a01, a11 = A[0, 1], A[1, 1]
    r12 = q00 * a01 + q10 * a11
    
    # q2 is the orthogonal vector to q1
    q01, q11 = -q10, q00
    r22 = np.abs(q01 * a01 + q11 * a11)
    
    Q = np.empty((2, 2))
    Q[0, 0], Q[1, 0] = q00, q10
    Q[0, 1], Q[1, 1] = q01, q11
    
    R = np.zeros((2, 2))
    R[0, 0], R[0, 1], R[1, 1] = r11, r12, r22
    
    return Q, R

"""Analytical 3x3 QR decomposition (Modified Gram-Schmidt)."""
# Optimized scalar math to avoid NumPy overhead for 3x3
def qr_3x3(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Pre-extract elements for speed
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
    v02, v12, v22 = a02 - r13*q00 - r23*q01, \
                    a12 - r13*q10 - r23*q11, \
                    a22 - r13*q20 - r23*q21
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


def integrate(
    f,
    dt: float,
    t_span: tuple[float, float],
    y0: ArrayLike,
    method: str = 'RK4'
):
    """
    Integrate an ODE system (state only) using a fixed time step.
    """
    if method != 'RK4':
        raise ValueError(f"Method {method} is not supported.")

    t_eval = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t_eval)
    y = np.asarray(y0, dtype=float)
    y_eval = np.zeros((n_steps,) + y.shape)

    # 1. Burn transients
    for _ in np.arange(0, t_span[0], dt):
        y = rk4_step(f, dt, y)

    # 2. Integration loop
    for i in range(n_steps):
        y_eval[i] = y
        y = rk4_step(f, dt, y)

    return y_eval


def integrate_variational(
    f,
    dt: float,
    t_span: tuple[float, float],
    y0: ArrayLike,
    Phi0: np.ndarray,
    qr_method: str = 'householder'
):
    """
    Integrate both state and variational equations to compute fundamental solutions.
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t_eval)
    
    y = np.asarray(y0, dtype=float)
    Phi = np.asarray(Phi0, dtype=float)
    dim = Phi.shape[0]

    # Determine QR function once
    qr_func = qr_2x2 if (qr_method == 'gram-schmidt' and dim == 2) else \
              qr_3x3 if (qr_method == 'gram-schmidt' and dim == 3) else \
              np.linalg.qr

    # 1. Burn transients (tracked)
    for _ in np.arange(0, t_span[0], dt):
        Q, _ = qr_func(Phi)
        y, Phi = rk4_step_variational(f, dt, y, Q)

    # 2. Main integration loop (tracked)
    y_eval = np.zeros((n_steps,) + y.shape)
    Phi_eval = np.zeros((n_steps,) + Phi.shape)
    Q_eval = np.zeros((n_steps,) + Phi.shape)
    R_eval = np.zeros((n_steps,) + Phi.shape)

    for i in range(n_steps):
        y_eval[i] = y
        Q, R = qr_func(Phi)
        Phi_eval[i], Q_eval[i], R_eval[i] = Phi, Q, R
        y, Phi = rk4_step_variational(f, dt, y, Q)

    return y_eval, Phi_eval, Q_eval, R_eval


def local_lyapunov_exponents(Q_history: np.ndarray, J_history: np.ndarray) -> np.ndarray:
    """
    Compute the local Lyapunov exponents from the continuous QR formulation.
    
    Formula: chi_i(t) = (Q^T(t) * J(t) * Q(t))_ii
    
    Parameters
    ----------
    Q_history : np.ndarray
        Orthogonal matrices from the QR decomposition history, shape (N, dim, dim).
    J_history : np.ndarray
        Jacobian matrices along the trajectory history, shape (N, dim, dim).
        
    Returns
    -------
    local_lyap : np.ndarray
        Time series of local Lyapunov exponents, shape (N, dim).
    """
    # Batched matrix multiplication: Q^T @ J @ Q
    # Q_history.transpose(0, 2, 1) gives the transpose of each Q matrix in the stack
    QTJQ = Q_history.transpose(0, 2, 1) @ J_history @ Q_history
    # Extract the diagonal elements for each time step
    return np.diagonal(QTJQ, axis1=1, axis2=2)


def continuous_qr_spectrum(Q_history: np.ndarray, J_history: np.ndarray) -> np.ndarray:
    """
    Compute the Lyapunov spectrum using the continuous QR formulation with a simple mean.
    
    Parameters
    ----------
    Q_history : np.ndarray
        Orthogonal matrices from the QR decomposition history, shape (N, dim, dim).
    J_history : np.ndarray
        Jacobian matrices along the trajectory history, shape (N, dim, dim).
        
    Returns
    -------
    spectrum : np.ndarray
        The calculated Lyapunov exponents, shape (dim,).
    """
    local_lyap = local_lyapunov_exponents(Q_history, J_history)
    return np.mean(local_lyap, axis=0)


def discrete_qr_spectrum(R_history: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute the Lyapunov spectrum from the discrete QR formulation (history of R matrices).
    
    Formula: lambda_i = 1/(N*dt) * sum(ln|R_ii|)
    
    Parameters
    ----------
    R_history : np.ndarray
        Stack of upper triangular matrices from QR decompositions, shape (N, dim, dim).
    dt : float
        Time step used in the integration.
    burn_in : int, optional
        Number of initial steps to discard as transients (default: 1000).
        
    Returns
    -------
    spectrum : np.ndarray
        The calculated Lyapunov exponents, shape (dim,).
    """
    # Extract the diagonal elements for each R matrix
    R_diag = np.diagonal(R_history, axis1=1, axis2=2)
    # Compute the mean log of the absolute diagonal elements, skipping burn-in
    return np.mean(np.log(np.abs(R_diag)), axis=0) / dt


def matrix_exponential_spectrum(
    J_history: np.ndarray, 
    dt: float, 
    qr_method: str = 'householder'
) -> np.ndarray:
    """
    Compute the Lyapunov spectrum using the Matrix Exponential formulation.
    
    Formula: Q_next, R_next = QR( exp(J*dt) @ Q_current )
    
    Parameters
    ----------
    J_history : np.ndarray
        Jacobian matrices along the trajectory, shape (N, dim, dim).
    dt : float
        Time step used in the integration.
    qr_method : str, optional
        QR decomposition method. 'householder' (default) or 'gram-schmidt'.
    burn_in : int, optional
        Number of initial steps to discard as transients (default: 1000).
        
    Returns
    -------
    spectrum : np.ndarray
        The calculated Lyapunov exponents, shape (dim,).
    """
    n_steps, dim, _ = J_history.shape
    Q = np.eye(dim)
    R_diags = np.zeros((n_steps, dim))
    
    # Determine QR function once
    qr_func = qr_2x2 if (qr_method == 'gram-schmidt' and dim == 2) else \
              qr_3x3 if (qr_method == 'gram-schmidt' and dim == 3) else \
              np.linalg.qr

    for i in range(n_steps):
        # Evolution of the tangent space via matrix exponential
        M = expm(dt * J_history[i])
        Q, R = qr_func(M @ Q)
        R_diags[i] = np.diagonal(R)
        
    # Return spectrum via mean log-diagonal of R
    return np.mean(np.log(np.abs(R_diags)), axis=0) / dt
