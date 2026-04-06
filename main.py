import numpy as np
import time
import sys
from pathlib import Path
from contextlib import contextmanager

# Ensure we can import from src directory
src_path = str(Path(__file__).parent.resolve() / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from le_calc.maps import LogisticMap, HenonMap
from le_calc.odes import Lorenz63, Rossler
from le_calc.methods import (
    continuous_qr_spectrum, 
    discrete_qr_spectrum,
    matrix_exponential_spectrum,
    taylor_spectrum
)

# Global constants for uniform output formatting
SPACING1 = 64  # For header separator lines
SPACING2 = 30  # For label widths in benchmarks

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
        print("-"*SPACING1 + f"\n      {name.upper()}\n" + "-"*SPACING1)
        
        # Phase 1: JIT Warmup
        with timer("JIT Warmup / Initialization"): 
            system = cls()
        
        # Phase 2: Trajectory Simulation
        with timer(f"Simulation (1,000,000 steps)"): 
            system.simulate(x0, 1_000_000)
        
        # Phase 3: Lyapunov Analysis
        qr_methods = ['householder', 'gram-schmidt'] if system.dim > 1 else ['householder']
        for qm in qr_methods:
            with timer(f"Discrete QR ({qm.capitalize()})"): 
                spec = system.discrete_qr_lyapunov_spectrum(qm)
            print_spectrum(f"Lyapunov Spectrum", spec)

def run_ode_benchmark(name, cls, x0, t_burn, t_window, dt):
    """Generic helper to run and compare Lyapunov methods for an ODE system."""
    print("\n" + "="*SPACING1 + f"\n      {name.upper()} BENCHMARKS\n" + "="*SPACING1)
    
    # Phase 1: Setup & Warmup
    with timer("JIT Warmup / Initialization"): 
        system = cls()
    
    Phi0 = np.eye(system.dim)
    t_span = (t_burn, t_burn + t_window)

    print(f"{'Transient period (Burn-in)':{SPACING2}}: {t_burn}")
    print(f"{'Integration window':{SPACING2}}: {t_window} (from t={t_span[0]} to {t_span[1]})")
    print(f"{'Time step (dt)':{SPACING2}}: {dt}")
    print(f"{'Number of steps':{SPACING2}}: {int(t_window / dt):,}")

    for method in ['RK2', 'RK4']:
        for qm in ['gram-schmidt', 'householder']:
            print("\n" + "-"*SPACING1 + f"\n      {qm.upper()} QR ({method})\n" + "-"*SPACING1)
            
            # Phase 2: Integration of Variational Equations
            with timer("Integration"):
                _, _, Q_hist, R_hist, _ = system.simulate_var(dt, t_span, x0, Phi0, method, qm)
            
            # Phase 3: Lyapunov Spectrum Comparison
            with timer("Method: Discrete QR"):
                spec1 = discrete_qr_spectrum(R_hist, dt)
            print_spectrum("Lyapunov Spectrum", spec1)
            
            with timer("Method: Continuous QR"):
                spec2 = continuous_qr_spectrum(Q_hist, system.J)
            print_spectrum("Lyapunov Spectrum", spec2)
            
            # with timer("Method: Matrix Exp (2nd)"):
            #     spec3 = matrix_exponential_spectrum(system.J, dt, qr_method=qm, order=2)
            # print_spectrum("Lyapunov Spectrum", spec3)
            
            with timer("Method: Matrix Exp (4th)"):
                spec4 = matrix_exponential_spectrum(system.J, dt, qr_method=qm, order=4)
            print_spectrum("Lyapunov Spectrum", spec4)
            
            # with timer("Method: Taylor Exp"):
            #     spec5 = taylor_spectrum(system.J, system.calc_xdot_H(), dt, qr_method=qm)
            # print_spectrum("Lyapunov Spectrum", spec5)

def run_lorenz_benchmarks():
    """Run benchmarks for the Lorenz 63 system."""
    run_ode_benchmark(
        name="Lorenz 63 ODE",
        cls=Lorenz63,
        x0=[1.0, 1.0, 10.0],
        t_burn=100.0,
        t_window=2500.0,
        dt=0.005
    )

def run_rossler_benchmarks():
    """Run benchmarks for the Rossler system."""
    run_ode_benchmark(
        name="Rossler ODE",
        cls=Rossler,
        x0=[1.0, 1.0, 0.5],
        t_burn=100.0,
        t_window=5000.0,
        dt=0.01
    )

if __name__ == "__main__":
    run_discrete_map_benchmarks()
    run_lorenz_benchmarks()
    run_rossler_benchmarks()
