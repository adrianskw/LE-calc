import numpy as np
import sys
import os

# Ensure we can import from src directory
if os.path.join(os.getcwd(), 'src') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'src'))

from le_calc.odes import Lorenz63
from le_calc.utils import integrate, integrate_variational

def test_methods():
    model = Lorenz63()
    x0 = [1.0, 1.0, 10.0]
    t_span = (0, 1)
    dt = 0.01
    
    print("Testing 'integrate' with different methods:")
    for m in ['RK1', 'RK2', 'RK3', 'RK4']:
        y = integrate(model, dt, t_span, x0, method=m)
        print(f"  {m:4}: Final point {y[-1]}")

    print("\nTesting 'integrate_variational' with different methods:")
    Phi0 = np.eye(3)
    for m in ['RK1', 'RK2', 'RK3', 'RK4']:
        y, Phi, Q, R = integrate_variational(model, dt, t_span, x0, Phi0, method=m)
        print(f"  {m:4}: Final point {y[-1]}")

if __name__ == "__main__":
    test_methods()
