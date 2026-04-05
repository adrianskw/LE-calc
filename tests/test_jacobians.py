import numpy as np
import pytest
from scipy.optimize import approx_fprime

def _check_analytical_jacobian(system, n_samples=5):
    """Helper to verify that system.jac(x) matches numerical gradients."""
    np.random.seed(42)
    
    # Use standard 1e-8 step size for finite differences
    epsilon = np.sqrt(np.finfo(float).eps)

    # Some systems (like maps) output multiple evaluations if x is 2D.
    # So we'll test one state at a time.
    for _ in range(n_samples):
        # Generate random state
        if system.dim == 1:
            x = np.random.rand(1) * 2 - 1
        else:
            x = np.random.randn(system.dim)

        if hasattr(system, 'ode'):
            f = system.ode
        else:
            def f(val):
                return system.forward_map(val)[0] if system.dim == 1 else system.forward_map(val)
            
        if system.dim == 1:
             # Specialized helper for 1D as approx_fprime behaves a bit differently
             def func_1d(val): return f(val)
             j_num = approx_fprime(x, func_1d, epsilon)
             j_ana = system.jac(x)
             # map jacobian is vectorized so it returns shape (n, 1, 1). extract it.
             if j_ana.ndim == 3: j_ana = j_ana[0, 0, 0]
        else:
            # For >1D we need to build the Jacobian row by row numerically
            j_num = np.zeros((system.dim, system.dim))
            for i in range(system.dim):
                def func_i(val, idx=i): return f(val)[idx]
                j_num[i, :] = approx_fprime(x, func_i, epsilon)
                
            j_ana = system.jac(x)
            if j_ana.ndim == 3: j_ana = j_ana[0] # Handle vectorized maps

        np.testing.assert_allclose(j_ana, j_num, rtol=1e-3, atol=1e-5)

def test_lorenz_jacobian(lorenz_system):
    _check_analytical_jacobian(lorenz_system)

def test_rossler_jacobian(rossler_system):
    _check_analytical_jacobian(rossler_system)

def test_logistic_jacobian(logistic_system):
    _check_analytical_jacobian(logistic_system)

def test_henon_jacobian(henon_system):
    _check_analytical_jacobian(henon_system)
