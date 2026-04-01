# Audit Report: `main.py` Logic & Correctness

## Overview
The script is structurally sound, runs without runtime errors, and accurately executes the algorithms necessary to determine Lyapunov exponents for Maps and ODEs. The implementations elegantly verify the mathematical equivalences of discrete and continuous methods for finding the Lyapunov spectrum. 

When observing the output of `main.py` locally, you can verify how well these different estimations agree:
```
Discrete QR every step, QR on Phi: [+0.90264 +0.00224 -14.57154]
Discrete QR every step, Saving R : [+0.90264 +0.00224 -14.57154]
Discrete QR Matrix Exponential   : [+0.90258 +0.00220 -14.57146]
Discrete QR Random Integrator    : [+1.07931 +0.22762 -15.18685]
Continuous QR                    : [+0.91266 -0.00114 -14.57818]
```

---

## Methodological Validation

### 1. Maps Validation
The `simulate()` and `discrete_qr_lyapunov_spectrum()` calls for `LogisticMap` and `HenonMap` correctly execute the chaotic orbits and output expected scalar (-D and 2-D) Lyapunov exponents. No logical flaws are present here.

### 2. Discarding Transients (`integrate`)
In `main.py`, the trajectory generation for the Lorenz system discards the initial `[0, 50]` simulation time window. 
- *Why this works:* During this initial phase, the fundamental matrix `Phi0` and its `Q` orthogonal basis naturally align with the theoretical Lyapunov vectors (the Oseledec bundle). The saved variable `Q_history[0]` at $t=50$ represents a fully matured tangent space basis.

### 3. Discrete QR on RK4 (`Phi` vs saved `R`)
- *Logic:* Computing `QR(Phi)` each step identically matches the operation `q, r = qr(Phi)` inside `integrate`. 
- *Correctness:* Your script elegantly proves they are perfectly identical. Extracting `R` directly during the `integrate` loop avoids redundant matrix factorizations, improving algorithmic efficiency.

### 4. Matrix Exponential & Forward Euler Methods
- *Logic:* The system uses analytic approximation matrices `expm(dt * J)` (exact assuming a piecewise-constant Jacobian) and `I + dt * J` (Forward Euler) applied to `Q` across each time step.
- *Correctness:* It demonstrates how these approximations approach the true value. The Forward Euler Integrator yields noticeably less accurate results (divergence sum is ~`-13.88` instead of ~`-13.66`), which mathematically confirms that first-order approximations fail to strictly preserve phase-space volume properties in chaotic regimes compared to the superior RK4 integrator.

### 5. Continuous QR Calculation
- *Logic:* Evaluates $\lim_{T \to \infty} \frac{1}{T} \int_0^T \text{diag}(Q^T J Q) dt$.
- *Correctness:* Mathematically correct implementation. It achieves nearly identical exponent outputs and identical mathematical trace constraints. The small numeric offset from discrete RK4 operations originates strictly from computing a zero-order Riemann integral average (`np.mean`), whereas the RK4 method inherently achieves 4th-order integral accumulation in `log(R)`. 

---

## Constructive Recommendations for `main.py` Improvement

> [!TIP]
> **Use initial tangent alignment rather than Re-Setting `Q`**
> For the Matrix Exponential and Forward Euler comparisons, the script arbitrarily resets `Q = np.eye(dim)`. This abruptly zeroes the bundle you worked to orient via the transient burn inside `integrate`. It takes multiple seconds of simulation time for vectors to re-align. 
> 
> ***Suggested fix***: Initialize with `Q = Q_history[0].copy()` instead.

> [!NOTE]
> **Harmonize Array Slicing over the Averages**
> The `logR_history` loops cut the first 1000 items (`logR_history[1000:]`), while `local_lyap` averages the entire array (`np.mean(local_lyap)`).
> Since your time history implicitly begins at $t=50$ fully matured, the `[1000:]` discard isn't explicitly necessary, but if keeping it out of an abundance of caution, you should align the comparison bounds.
>
> ***Suggested fix***:
> ```python
> np.mean(local_lyap[1000:], axis=0) # Match slicing intervals
> ```

> [!WARNING]
> **Higher-Order Numerical Integration of Continuous Exponents**
> The continuous operator calculates raw rates per step (`Q.T @ J @ Q`) and averages them directly via `np.mean`. To match the precise numerical fidelity of Discrete QR's 4th-order method, continuous rates should be numerically integrated across time using schemes like Simpson's rule or Trapezoid arrays, rather than a raw arithmetic mean.
