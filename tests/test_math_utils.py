import numpy as np
import pytest
from le_calc.utils import qr_GS_2x2, qr_GS_3x3, qr_HH

def test_qr_gs_2x2():
    np.random.seed(42)
    A = np.random.randn(2, 2)
    Q, R = qr_GS_2x2(A)
    
    # 1. Orthonormality: Q^T @ Q ≈ I
    np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-10)
    
    # 2. Upper triangular R
    assert np.abs(R[1, 0]) < 1e-10
    
    # 3. Reconstructs A
    np.testing.assert_allclose(Q @ R, A, atol=1e-10)

def test_qr_gs_3x3():
    np.random.seed(42)
    A = np.random.randn(3, 3)
    Q, R = qr_GS_3x3(A)
    
    # 1. Orthonormality
    np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)
    
    # 2. Upper triangular R
    assert np.abs(R[1, 0]) < 1e-10
    assert np.abs(R[2, 0]) < 1e-10
    assert np.abs(R[2, 1]) < 1e-10
    
    # 3. Reconstructs A
    np.testing.assert_allclose(Q @ R, A, atol=1e-10)

def test_qr_match_numpy():
    np.random.seed(42)
    A3 = np.random.randn(3, 3)
    
    Q_gs, R_gs = qr_GS_3x3(A3)
    Q_np, R_np = qr_HH(A3)
    
    # QR factorization is unique up to signs of columns of Q (and rows of R)
    # We can check that absolute values match
    np.testing.assert_allclose(np.abs(Q_gs), np.abs(Q_np), atol=1e-10)
    np.testing.assert_allclose(np.abs(R_gs), np.abs(R_np), atol=1e-10)
