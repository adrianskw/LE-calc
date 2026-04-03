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

# Global constants for uniform output formatting
SPACING1 = 66  # For header separator lines
SPACING2 = 33  # For label widths in benchmarks

@contextmanager
def timer(label):
    """Simple context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{label:{SPACING2}}: {elapsed:.4f}s")

def print_spectrum(label, spectrum):
    """Formatted print for Lyapunov spectrum results."""
    print(f"{label:{SPACING2}}: {np.array2string(spectrum, formatter={'float_kind':lambda x: f'{x:+.5f}'})}")

def run_discrete_map_benchmarks():
    """Run simulation and spectrum benchmarks for discrete-time maps."""
    print("\n" + "="*SPACING1 + "\n      DISCRETE MAP BENCHMARKS\n" + "="*SPACING1)
    
    for name, cls, x0 in [("Logistic Map (r=4.0)", LogisticMap, 0.65), 
                          ("Henon Map (a=1.4, b=0.3)", HenonMap, [0.5, 0.2])]:
        print("\n" + "-"*SPACING1 + f"\n      {name.upper()}\n" + "-"*SPACING1)
        
        # Phase 1: JIT Warmup
        with timer("JIT Warmup / Initialization"): 
            sys = cls()
        
        # Phase 2: Trajectory Simulation
        with timer(f"Simulation (1,000,000 steps)"): 
            sys.simulate(x0, 1_000_000)
        
        # Phase 3: Lyapunov Analysis
        qr_methods = ['householder', 'gram-schmidt'] if sys.dim > 1 else ['householder']
        for qm in qr_methods:
            with timer(f"Calculation Time"): 
                spec = sys.discrete_qr_lyapunov_spectrum(qm)
            print_spectrum(f"Lyapunov Spectrum ({qm})", spec)

def run_lorenz_benchmarks():
    """Run and compare different Lyapunov estimation methods for the Lorenz ODE."""
    print("\n" + "="*SPACING1 + "\n      LORENZ 63 ODE BENCHMARKS\n" + "="*SPACING1)
    
    # Phase 1: Setup & Warmup
    with timer("JIT Warmup / Initialization"): 
        system = Lorenz63()
    
    x0, Phi0, dt = [1.0, 1.0, 10.0], np.eye(3), 0.0025
    t_burn, t_window = 50.0, 2500.0
    t_span = (t_burn, t_burn + t_window)

    print(f"{'Transient period (Burn-in)':{SPACING2}}: {t_burn}")
    print(f"{'Integration window':{SPACING2}}: {t_window} (from t={t_span[0]} to {t_span[1]})")
    print(f"{'Time step (dt)':{SPACING2}}: {dt}")
    print(f"{'Number of steps':{SPACING2}}: {int(t_window / dt):,}")

    for method in ['RK2', 'RK4']:
        
        for qm in ['gram-schmidt', 'householder']:
            print("\n" + "-"*SPACING1 + f"\n      {qm.upper()} QR ({method})\n" + "-"*SPACING1)
            
            # Phase 2: Integration of Variational Equations
            with timer(f"Integration ({method}/{qm.capitalize()})"):
                _, _, Q_hist, R_hist = system.simulate_var(dt, t_span, x0, Phi0, method, qm)
            
            # Phase 3: Lyapunov Spectrum Comparison
            with timer("Method 1: Discrete QR"):
                spec1 = discrete_qr_spectrum(R_hist, dt)
            print_spectrum("Lyapunov Spectrum", spec1)
            
            with timer("Method 2: Continuous QR"):
                spec2 = continuous_qr_spectrum(Q_hist, system.calc_jac())
            print_spectrum("Lyapunov Spectrum", spec2)
            
            with timer("Method 3: Matrix Exp"):
                spec3 = matrix_exponential_spectrum(system.J, dt, qr_method=qm)
            print_spectrum("Lyapunov Spectrum", spec3)

if __name__ == "__main__":
    run_discrete_map_benchmarks()
    run_lorenz_benchmarks()
