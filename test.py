from fpylll import IntegerMatrix, LLL, FPLLL
from copy import copy
import time

FPLLL.set_random_seed(42)

def partial_lll(A: IntegerMatrix, k: int):
    """
    Divide and Assemble LLL reduced submatrix.
    Given a square matrix of A (dim = n), apply LLL on each k-th submatrix of size (n/k, n)
    then combine reduced bases for each submatrix into a partially reduced matrix of size (n, n)
    """
    assert A.ncols == A.nrows, "A has to be a square IntegerMatrix"
    n = A.ncols
    assert n % k == 0, "choose a partial divide factor that divides dimension n"
    aggregated = IntegerMatrix(n, n)
    for idx in range(k):
        sub_A = A.submatrix(idx * n/k, 0, (idx+1) * n/k, n)
        reduced_sub_A = LLL.reduction(sub_A)
        for i in range(n//k):
            for j in range(n):
                aggregated[idx * n/k + i, j] = reduced_sub_A[i, j]
    return aggregated


print("dim n, ratio(2), ratio(4), ratio(8)")

for n in range(40, 401, 40):
    num_samples = 30

    total_ratio_two = 0
    total_ratio_four = 0
    total_ratio_eight = 0

    total_time_two = 0
    total_time_four = 0
    total_time_eight = 0

    for _ in range(num_samples):
        A = IntegerMatrix.random(n, "uniform", bits=8)
        original = copy(A)
        assert not LLL.is_reduced(A)

        reduced_A = LLL.reduction(A)
        assert LLL.is_reduced(reduced_A)
        norm_fully_reduced = A[0].norm()

        # split in 2 parts
        A = copy(original)
        start_time = time.time()
        aggregated_two = partial_lll(A, 2)
        total_time_two += time.time() - start_time
        assert A == original

        start_time = time.time()
        aggregated_four = partial_lll(A, 4)
        total_time_four += time.time() - start_time
        assert A == original

        start_time = time.time()
        aggregated_eight = partial_lll(A, 8)
        total_time_eight += time.time() - start_time
        assert A == original

        total_ratio_two += aggregated_two[0].norm() / norm_fully_reduced
        total_ratio_four += aggregated_four[0].norm() / norm_fully_reduced
        total_ratio_eight += aggregated_eight[0].norm() / norm_fully_reduced

    # print(f"{n}, {total_ratio_two/num_samples}")
    print(
        f"{n}, "
        f"{total_ratio_two/num_samples:.6f} ({total_time_two/num_samples * 1000:.3f} ms), "
        f"{total_ratio_four/num_samples:.6f} ({total_time_four/num_samples * 1000:.3f} ms), "
        f"{total_ratio_eight/num_samples:.6f} ({total_time_eight/num_samples * 1000:.3f} ms)"
    )
