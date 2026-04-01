import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import from src directory
import sys
import os
if os.path.join(os.getcwd(), 'src') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'src'))


from le_calc.maps import LogisticMap

# Initialize Logistic Map (default r = 4.0, chaotic)
model = LogisticMap()
x0=0.65
model.simulate(x0,n_steps=50000)

# calculating Lyapunov exponents
model.discrete_qr_lyapunov_spectrum()


from le_calc.maps import HenonMap

# Initialize chaotic Hénon Map (default a=1.4, b=0.3)
model = HenonMap()
x0 = [0.5, 0.2]  
model.simulate(x0, n_steps=50000)
# calculating Lyapunov exponents
model.discrete_qr_lyapunov_spectrum()

from le_calc.odes import Lorenz63
from le_calc.utils import integrate

# Initialize Lorenz system
model = Lorenz63()

# Define initial conditions and time span
x0 = [1.0, 1.0, 10.0]
dim = len(x0)
Phi0 = np.eye(dim)
t_span = (50, 350)
delta_t = 0.005
n_steps = int((t_span[1]-t_span[0])/delta_t)

# integrate discarding transients
x_history,Phi,Q_history,R_history = integrate(model, delta_t, t_span,x0,Phi0)
# x_history = integrate(model, delta_t, t_span,x0)

from scipy.linalg import expm

logR_history = np.zeros((n_steps,dim))
for i in range(n_steps):
    Q,R = np.linalg.qr(Phi[i])
    logR_history[i] = np.log(np.abs(np.diag(R)))
print(f"Discrete QR every step, QR on Phi: {np.array2string(np.mean(logR_history[1000:],axis=0)/delta_t, formatter={'float_kind':lambda x: f'{x:+.5f}'})}")

logR_history = np.zeros((n_steps,dim))
for i in range(n_steps):
    logR_history[i] = np.log(np.abs(np.diag(R_history[i])))
print(f"Discrete QR every step, Saving R : {np.array2string(np.mean(logR_history[1000:],axis=0)/delta_t, formatter={'float_kind':lambda x: f'{x:+.5f}'})}")

Q = np.eye(dim)
logR_history = np.zeros((n_steps,dim))
for i in range(n_steps):
    Q,R = np.linalg.qr(expm(delta_t*model.jac(x_history[i]))@Q)
    logR_history[i] = np.log(np.abs(np.diag(R)))
print(f"Discrete QR Matrix Exponential   : {np.array2string(np.mean(logR_history[1000:],axis=0)/delta_t, formatter={'float_kind':lambda x: f'{x:+.5f}'})}")

def integrator(x):
    return np.eye(dim)+delta_t*model.jac(x)

Q = np.eye(dim)
logR_history = np.zeros((n_steps,dim))
for i in range(n_steps):
    Q,R = np.linalg.qr(integrator(x_history[i])@Q)
    logR_history[i] = np.log(np.abs(np.diag(R)))
print(f"Discrete QR Random Integrator    : {np.array2string(np.mean(logR_history[1000:],axis=0)/delta_t, formatter={'float_kind':lambda x: f'{x:+.5f}'})}")

local_lyap = np.ones(x_history.shape)
for i in range(n_steps):
    local_lyap[i] = np.diag(Q_history[i].T@model.jac(x_history[i])@Q_history[i])
print(f"Continuous QR                    : {np.array2string(np.mean(local_lyap,axis=0), formatter={'float_kind':lambda x: f'{x:+.5f}'})}")