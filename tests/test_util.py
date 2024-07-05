# -*- coding: utf-8 -*-

from fpylll import IntegerMatrix, GSO
from fpylll.util import adjust_radius_to_gh_bound, set_random_seed, gaussian_heuristic, to_mpz
import numpy as np
import gmpy2
import time

dimensions = [20, 21, 40, 41, 60, 61, 80, 81, 100, 101, 200, 201, 300, 301, 400, 401]


def make_integer_matrix(n):
    A = IntegerMatrix.random(n, "uniform", bits=30)
    return A


def test_gh():
    try:
        from fpylll.numpy import dump_r
    except ImportError:
        return

    for n in dimensions:
        set_random_seed(n)
        A = make_integer_matrix(n)
        try:
            M = GSO.Mat(A, float_type="ld")
        except ValueError:
            M = GSO.Mat(A, float_type="d")
        M.discover_all_rows()
        M.update_gso()
        radius = M.get_r(0, 0)
        root_det = M.get_root_det(0, n)
        gh_radius, ge = adjust_radius_to_gh_bound(2000*radius, 0, n, root_det, 1.0)

        gh1 = gh_radius * 2**ge

        r = dump_r(M, 0, n)
        gh2 = gaussian_heuristic(r)
        assert abs(gh1/gh2 -1) < 0.01

def test_scale_float_to_mpz():
    # first ensure correctness
    a32 = np.float32(np.random.rand())
    assert a32.dtype == np.float32
    assert np.float32(np.float64(a32)) == a32
    assert gmpy2.cmp(gmpy2.mpfr(to_mpz(a32)), gmpy2.mpfr(str(float(a32))) * 2**150) == 0

    a64 = np.float64(np.random.rand())
    assert a64.dtype == np.float64
    assert gmpy2.cmp(gmpy2.mpfr(to_mpz(a64)), gmpy2.mpfr(str(a64)) * 2**1075) == 0

    # now run some benchmark
    n = 1000000
    print(f"\nBenchmarking for {n} calls")

    start = time.perf_counter()
    for _ in range(n):
        _ = gmpy2.mpfr(str(float(a32))) * 2**150
    end = time.perf_counter()
    print(f"gmpy2 takes: {end - start} sec")

    start = time.perf_counter()
    for _ in range(n):
        _ = to_mpz(a32)
    end = time.perf_counter()
    print(f"to_mpz takes: {end - start} sec")
