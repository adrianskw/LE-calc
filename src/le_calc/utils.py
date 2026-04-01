"""
utils.py — Shared utilities for ODE integration and dynamical systems analysis.
"""

import numpy as np
from numpy.typing import ArrayLike


def rk4_step(model, dt: float, y: np.ndarray, Phi=None) -> np.ndarray:
    """Take a single Runge-Kutta 4th order step."""
    k1 = model.ode(y)
    k2 = model.ode(y + (dt / 2.0) * k1)
    k3 = model.ode(y + (dt / 2.0) * k2)
    k4 = model.ode(y + dt * k3)

    if Phi is None:
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        # Variational equation: dPhi/dt = J(x(t)) @ Phi
        # RK4 stages for Phi, using Jacobian at the same sub-steps as y.
        L1 = model.jac(y)                   @  Phi
        L2 = model.jac(y + (dt / 2.0) * k1) @ (Phi + (dt / 2.0) * L1)
        L3 = model.jac(y + (dt / 2.0) * k2) @ (Phi + (dt / 2.0) * L2)
        L4 = model.jac(y +  dt        * k3) @ (Phi +  dt        * L3)
        return [y   + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
                Phi + (dt / 6.0) * (L1 + 2.0 * L2 + 2.0 * L3 + L4)]


def integrate(
    f,
    delta_t: float,
    t_span: tuple[float, float],
    y0: ArrayLike,
    Phi0: np.ndarray = None,
    method: str = 'RK4'
):
    """Integrate an ODE system using a fixed time step.

    Parameters
    ----------
    f : object
        The model object containing .ode(y) and optionally .jac(y).
    delta_t : float
        Fixed time step.
    t_span : tuple
        (start_time, end_time) to integrate.
    y0 : array-like
        Initial condition.
    Phi0 : np.ndarray, optional
        Initial fundamental solution matrix. If True, initializes as identity.
    method : str
        Integration method.

    Returns
    -------
    y_eval : np.ndarray
        Time series array.
    phi_eval : np.ndarray, optional
        Fundamental solution matrix series if Phi0 is provided.
    """
    step_funcs = {
        'RK4': rk4_step,
    }

    if method not in step_funcs:
        raise ValueError(f"Method {method} is not supported.")

    step_func = step_funcs[method]
    t_eval = np.arange(t_span[0], t_span[1], delta_t)
    n_steps = len(t_eval)
    
    # convert to numpy arrays
    y = np.asarray(y0, dtype=float)
    n = y.size

    # burn transients
    for _ in np.arange(0, t_span[0], delta_t):
        # not saving y
        if Phi0 is not None:
            Q,R = np.linalg.qr(Phi0)
            # not saving Phi
        else:
            Q = None
        
        res = step_func(f, delta_t, y, Q)
        # updating
        if Phi0 is not None:
            y, Phi0 = res
        else:
            y = res

    y_eval = np.zeros((n_steps, *y.shape))
    if Phi0 is not None:
        Phi_eval = np.zeros((n_steps, *Phi0.shape))
        Q_eval = np.zeros((n_steps, *Phi0.shape))
        R_eval = np.zeros((n_steps, *Phi0.shape))

    for i in range(n_steps):
        y_eval[i] = y # save y
        if Phi0 is not None:
            Q,R = np.linalg.qr(Phi0)
            Phi_eval[i] = Phi0 # save Phi
            Q_eval[i] = Q # save Q to do continuous QR
            R_eval[i] = R # save R to not do a QR outside this loop again
        else:
            Q = None

        res = step_func(f, delta_t, y, Q)
    
        # updating
        if Phi0 is not None:
            y, Phi0 = res
        else:
            y = res

    if Phi0 is not None:
        return y_eval, Phi_eval, Q_eval, R_eval
    return y_eval
