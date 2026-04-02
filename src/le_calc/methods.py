"""
methods.py — methods for calculating Lyapunov exponents
"""

import numpy as np
from scipy.linalg import expm
from .utils import qr_GS_2x2, qr_GS_3x3, qr_HH


def local_lyapunov_exponents(Q_history: np.ndarray, J_history: np.ndarray) -> np.ndarray:
    """
    Compute the local Lyapunov exponents from the continuous QR formulation.
    Formula: chi_i(t) = (Q^T(t) * J(t) * Q(t))_ii
    """
    # Batched matrix multiplication: Q^T @ J @ Q
    # Q_history.transpose(0, 2, 1) gives the transpose of each Q matrix in the stack
    QTJQ = Q_history.transpose(0, 2, 1) @ J_history @ Q_history
    # Extract the diagonal elements for each time step
    return np.diagonal(QTJQ, axis1=1, axis2=2)


def continuous_qr_spectrum(Q_history: np.ndarray, J_history: np.ndarray) -> np.ndarray:
    """Compute the Lyapunov spectrum using the continuous QR formulation with a simple mean."""
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
        Time step.
    qr_method : str
        QR decomposition method.

    Returns
    -------
    spectrum : np.ndarray
        The calculated Lyapunov exponents, shape (dim,).
    """
    n_steps, dim, _ = J_history.shape
    Q = np.eye(dim)
    R_diags = np.zeros((n_steps, dim))
    
    qr_func = qr_GS_2x2 if (qr_method == 'gram-schmidt' and dim == 2) else \
              qr_GS_3x3 if (qr_method == 'gram-schmidt' and dim == 3) else \
              qr_HH

    for i in range(n_steps):
        # Evolution of the tangent space via matrix exponential
        M = expm(dt * J_history[i])
        Q, R = qr_func(M @ Q)
        R_diags[i] = np.diagonal(R)
        
    # Return spectrum via mean log-diagonal of R
    return np.mean(np.log(np.abs(R_diags)), axis=0) / dt
