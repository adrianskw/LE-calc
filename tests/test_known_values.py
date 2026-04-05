import numpy as np
import pytest

def test_logistic_map_known_value(logistic_system):
    # Logistic map at r=4 is fully chaotic with Lyapunov exponent ln(2)
    logistic_system.r = 4.0
    logistic_system.simulate(0.65, 50000, 1000)
    spec = logistic_system.discrete_qr_lyapunov_spectrum('householder')
    
    ln2 = np.log(2)
    np.testing.assert_allclose(spec[0], ln2, rtol=1e-2, atol=1e-2)

def test_henon_map_known_value(henon_system):
    # Henon map at a=1.4, b=0.3
    # The sum of exponents should equal ln(0.3) due to constant area contraction
    henon_system.simulate(np.array([0.5, 0.2]), 50000, 1000)
    spec = henon_system.discrete_qr_lyapunov_spectrum('householder')
    
    expected_sum = np.log(0.3)
    np.testing.assert_allclose(np.sum(spec), expected_sum, rtol=1e-3, atol=1e-3)

def test_lorenz_known_value(lorenz_system):
    # Standard Lorenz parameters sigma=10, rho=28, beta=8/3
    # Known spectrum is approx [0.906, 0.0, -14.572]
    # We use a relatively short integration so we use relaxed tolerances
    
    dt = 0.01
    x0 = np.array([1.0, 1.0, 10.0])
    Phi0 = np.eye(3)
    
    _, _, Q_hist, _, _ = lorenz_system.simulate_var(
        dt=dt, 
        t_span=(50.0, 50.0 + 500.0), # 500 seconds integration
        x0=x0, 
        Phi0=Phi0, 
        method='RK4', 
        qr_method='householder'
    )
    
    from le_calc.methods import continuous_qr_spectrum
    spec = continuous_qr_spectrum(Q_hist, lorenz_system.J)
    
    # Sort descending
    spec = np.sort(spec)[::-1]
    
    expected = [0.906, 0.0, -14.572]
    np.testing.assert_allclose(spec, expected, rtol=5e-2, atol=5e-2)
    
