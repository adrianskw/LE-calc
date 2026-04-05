import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure src is in the python path
src_path = str(Path(__file__).parent.parent.resolve() / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from le_calc.odes import Lorenz63, Rossler
from le_calc.maps import LogisticMap, HenonMap

@pytest.fixture
def lorenz_system():
    return Lorenz63()

@pytest.fixture
def rossler_system():
    return Rossler()

@pytest.fixture
def logistic_system():
    return LogisticMap()

@pytest.fixture
def henon_system():
    return HenonMap()
