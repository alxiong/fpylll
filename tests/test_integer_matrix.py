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
