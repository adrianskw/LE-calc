"""
le_calc — Lyapunov Exponent Calculator
=======================================
Compute Lyapunov exponents from:
  - Known dynamical systems (ODEs / discrete maps): `le_calc.equations`
  - Empirical time-series data:                     `le_calc.data`

Shared utilities (plotting, synthetic datasets):    `le_calc.utils`
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("le-calc")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
