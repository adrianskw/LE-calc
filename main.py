import numpy as np
import time
import sys
import os
from contextlib import contextmanager

# Ensure we can import from src directory
if os.path.join(os.getcwd(), 'src') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'src'))

from le_calc.maps import LogisticMap, HenonMap
from le_calc.odes import Lorenz63
from le_calc.methods import (
    continuous_qr_spectrum, 
    discrete_qr_spectrum,
    matrix_exponential_spectrum
)

@contextmanager
def timer(label):
    """Simple context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{label:40}: {elapsed:.4f}s")

def print_spectrum(label, spectrum):
    """Formatted print for Lyapunov spectrum results."""
    print(f"{label:40}: {np.array2string(spectrum, formatter={'float_kind':lambda x: f'{x:+.5f}'})}")

def run_discrete_map_benchmarks():
    """Run simulation and spectrum benchmarks for discrete-time maps."""
    print("\n" + "="*50)
    print("      DISCRETE MAP BENCHMARKS")
    print("="*50)
    
    # 1. Logistic Map
    print("\n--- Logistic Map (r=4.0) ---")
    system = LogisticMap()
    x0 = 0.65
    n_steps=100000
    with timer(f"Simulation ({n_steps:,} steps)"):
        x_history = system.simulate(x0, n_steps=n_steps) # x_history unused, can be plotted
    with timer("Spectrum Calculation (Discrete QR)"):
        spec = system.discrete_qr_lyapunov_spectrum() # uses x_history stored in system object
    print_spectrum("Logistic Map Results", spec)

    # 2. Henon Map
    print("\n--- Henon Map (a=1.4, b=0.3) ---")
    system = HenonMap()
    x0 = [0.5, 0.2]
    n_steps=100000
    with timer(f"Simulation ({n_steps:,} steps)"):
        x_history = system.simulate(x0, n_steps=n_steps) # x_history unused, can be plotted
    with timer("Spectrum Calculation (Householder)"):
        spec = system.discrete_qr_lyapunov_spectrum(qr_method='householder') # uses x_history stored in system object
    print_spectrum("Henon Results (Householder)", spec)
    with timer("Spectrum Calculation (Gram-Schmidt)"):
        spec_fast = system.discrete_qr_lyapunov_spectrum(qr_method='gram-schmidt') # uses x_history stored in system object
    print_spectrum("Henon Results (Gram-Schmidt)", spec_fast)

def run_lorenz_benchmarks():
    """Run and compare different Lyapunov estimation methods for the Lorenz ODE."""
    print("\n" + "="*50)
    print("      LORENZ 63 ODE BENCHMARKS")
    print("="*50)
    
    system = Lorenz63()
    x0 = [1.0, 1.0, 10.0]
    dim = len(x0)
    Phi0 = np.eye(dim)
    t_burn = 50.0
    t_window = 500.0
    t_span = (t_burn, t_burn + t_window)
    dt = 0.005
    n_steps = int(t_window / dt)

    print(f"Transient period (Burn-in) : {t_burn}")
    print(f"Integration window         : {t_window} (from t={t_span[0]} to {t_span[1]})")
    print(f"Time step                  : {dt}")
    print(f"Number of steps            : {n_steps:,}")

    # 1. Compare RK methods and QR Integration Methods
    for method in ['RK2', 'RK4']:
        print(f"\n{'='*15} Testing {method} Integration {'='*15}")
        
        for qr_method in ['gram-schmidt', 'householder']:
            print(f"\n--- {qr_method.capitalize()} QR ({method}) ---")
            
            with timer(f"Integration ({method}/{qr_method})"):
                x_history, Phi_history, Q_history, R_history = system.simulate_var(
                    dt, t_span, x0, Phi0, method=method, qr_method=qr_method
                )

            # Method 1: Discrete QR Result (extracted from R_history)
            # This method accumulates the diagonal elements of the upper-triangular R matrices
            # generated during the re-orthonormalized integration process.
            with timer("Method 1: Discrete QR (from R)"):
                spec1 = discrete_qr_spectrum(R_history, dt)
            print_spectrum("Discrete QR Spectrum", spec1)

            # Method 2: Continuous QR Result (computed from Q_history)
            # This method uses the analytical formula for local Lyapunov exponents:
            # chi_i(t) = (Q^T(t) * J(t) * Q(t))_ii and averages them.
            with timer("Method 2: Continuous QR (Mean)"):
                J_history = system.calc_jac() # uses x history stored in system object
                spec2 = continuous_qr_spectrum(Q_history, J_history)
            print_spectrum("Continuous QR Spectrum", spec2)

            # Method 3: Matrix Exponential
            # This method evolves the tangent space using the matrix exponential of the 
            # Jacobian (e^{J*dt}) and then performs QR re-orthonormalization.
            with timer("Method 3: Matrix Exponential"):
                    # Re-using J_history from the continuous QR step (last iteration)
                    spec3 = matrix_exponential_spectrum(J_history, dt, qr_method=qr_method)
            print_spectrum("Matrix Exp + Discrete QR Spectrum ", spec3)

if __name__ == "__main__":
    run_discrete_map_benchmarks()
    # run_lorenz_benchmarks()
