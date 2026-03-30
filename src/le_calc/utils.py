"""
utils.py — Shared utilities: plotting and integration helpers.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp


def integrate_ode(
    system,
    x0: ArrayLike,
    t_span: tuple[float, float],
    delta_t: float = 0.01,
    method: str = 'RK45',
):
    """Integrate an ODE system using a fixed time step.

    Parameters
    ----------
    system : object
        A dynamical system object that provides a callable ``f(t, x)``.
    x0 : array-like
        Initial condition.
    t_span : tuple
        (start_time, end_time) to integrate.
    delta_t : float
        Evaluation time step.
    method : str
        Integration method (default: 'RK45').

    Returns
    -------
    sol : scipy.integrate.OdeResult
        The solution object returned by ``solve_ivp``. 
        Trajectory is accessed via ``sol.y``.
    """
    t_eval = np.arange(t_span[0], t_span[1], delta_t)
    return solve_ivp(system.f, t_span, np.asarray(x0), t_eval=t_eval, method=method)
