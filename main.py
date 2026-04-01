import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from contextlib import contextmanager
from scipy.linalg import expm
from scipy.integrate import simpson

# Ensure we can import from src directory
if os.path.join(os.getcwd(), 'src') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'src'))

from le_calc.maps import LogisticMap, HenonMap
from le_calc.odes import Lorenz63
from le_calc.utils import (
    integrate, 
    integrate_variational, 
    continuous_qr_spectrum, 
    local_lyapunov_exponents,
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
    model = LogisticMap()
    x0 = 0.65
    with timer("Simulation (50,000 steps)"):
        model.simulate(x0, n_steps=50000)
    with timer("Spectrum Calculation (Discrete QR)"):
        spec = model.discrete_qr_lyapunov_spectrum()
    print_spectrum("Logistic Map Results", spec)

    # 2. Henon Map
    print("\n--- Henon Map (a=1.4, b=0.3) ---")
    model = HenonMap()
    x0 = [0.5, 0.2]
    with timer("Simulation (50,000 steps)"):
        model.simulate(x0, n_steps=50000)
    with timer("Spectrum Calculation (Householder)"):
        spec = model.discrete_qr_lyapunov_spectrum(qr_method='householder')
    print_spectrum("Henon Results (Householder)", spec)
    with timer("Spectrum Calculation (Gram-Schmidt)"):
        spec_fast = model.discrete_qr_lyapunov_spectrum(qr_method='gram-schmidt')
    print_spectrum("Henon Results (Gram-Schmidt)", spec_fast)

def run_lorenz_benchmarks():
    """Run and compare different Lyapunov estimation methods for the Lorenz ODE."""
    print("\n" + "="*50)
    print("      LORENZ 63 ODE BENCHMARKS")
    print("="*50)
    
    model = Lorenz63()
    x0 = [1.0, 1.0, 10.0]
    dim = len(x0)
    Phi0 = np.eye(dim)
    t_span = (50, 350) # burn in for 0<t<50
    dt = 0.005
    n_steps = int((t_span[1] - t_span[0]) / dt)

    # 1. Compare QR Integration Methods
    for qr_method in ['gram-schmidt', 'householder']:
        print(f"\n--- {qr_method.capitalize()} QR ---")
        
        with timer(f"RK4 Integration ({qr_method})"):
            x_history, Phi_history, Q_history, R_history = integrate_variational(
                model, dt, t_span, x0, Phi0, qr_method=qr_method
            )

        # Method 1: Discrete QR Result (extracted from R_history)
        with timer("Method 1: Discrete QR (from R)"):
            spec1 = discrete_qr_spectrum(R_history, dt)
        print_spectrum("Spectrum (Discrete QR)", spec1)

        # Method 2: Continuous QR Result (computed from Q_history)
        with timer("Method 2: Continuous QR (Mean)"):
            J_history = model.jac(x_history)
            spec2 = continuous_qr_spectrum(Q_history, J_history)
        print_spectrum("Spectrum (Continuous QR)", spec2)

        with timer("Matrix Exponential (expm)"):
            # Re-using J_history from the continuous QR step (last iteration)
            spec3 = matrix_exponential_spectrum(J_history, dt, qr_method=qr_method)
        print_spectrum("Method 3: Matrix Exp", spec3)    

if __name__ == "__main__":
    run_discrete_map_benchmarks()
    run_lorenz_benchmarks()
