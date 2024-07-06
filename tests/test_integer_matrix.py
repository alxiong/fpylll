import numpy as np
import gmpy2
from fpylll import IntegerMatrix

def test_accept_ndarray():
    nrows = 10
    ncols = 20
    a = np.random.rand(nrows, ncols)
    assert a.dtype == np.float64
    A = IntegerMatrix.from_matrix(a)
    for i in range(nrows):
        for j in range(ncols):
            assert A[i, j] == int(gmpy2.mpfr(a[i, j]) * 2**1075)

    b = np.astype(a, np.float32)
    assert b.dtype == np.float32
    B = IntegerMatrix.from_matrix(b)
    for i in range(nrows):
        for j in range(ncols):
            assert B[i, j] == int(gmpy2.mpfr(float(b[i, j])) * 2**150)

def test_set_submatrix():
    A_rows = 4
    A_cols = 5
    M_dim = 6
    A = np.random.randint(10, size=(A_rows, A_cols))
    M = IntegerMatrix(M_dim, M_dim)
    M.set_matrix(A, start_row = 1, start_col = 1)
    for i in range(A_rows):
        for j in range(A_cols):
            assert A[i, j] == M[i+1, j+1]

    B = IntegerMatrix.random(A_rows, "uniform", bits=30)
    M.set_matrix(B, start_row = 2, start_col = 2)
    for i in range(A_rows):
        for j in range(A_rows):
            assert B[i, j] == M[i+2, j+2]
