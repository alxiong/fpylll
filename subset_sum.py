# Module for solving Subset Sum Problem using LLL

import unittest
import numpy as np
from fpylll import LLL, IntegerMatrix


class SubsetSum:
    """
    Class for defining and solving Subset Sum problem
    """

    def __init__(self, multiset: np.ndarray, target: np.ndarray):
        """
        Define the problem instance with multiset {a_i} and target S
        """
        assert multiset.dtype == target.dtype
        assert multiset.ndim == target.ndim + 1
        assert multiset.ndim in [1, 2]

        if target.ndim == 0:
            multiset = np.expand_dims(multiset, axis=1)
            target = np.expand_dims(target, axis=0)
        self.multiset = multiset
        self.target = target

        # embedding dim: m
        self.embed_dim = target.shape[0]
        # lattice dim: n
        self.lattice_dim = multiset.shape[0]

        # d = n / log max a_i
        self.density = np.full(shape=target.shape, fill_value=self.lattice_dim) / float(
            np.log2(np.max(self.multiset))
        )
        print(
            f"lattice dim: {self.lattice_dim}, embedding dim: {self.embed_dim}, density: {self.density}"
        )

    def solve(self):
        """
        Solve the instance, return an array of value: `z in {0, 1}^self.lattice_dim`
        Such that `A*z = S`
        Read more: https://hackmd.io/@alxiong/ssp-from-lll
        """

        # TODO: refine this heuristic, theory bound only say N>\sqrt n
        N = int(10 * np.sqrt(self.lattice_dim))
        solutions = []

        # construct the bases matrix B which is (n+1) by (n+m+1) matrix:
        rows = self.lattice_dim + 1
        cols = self.lattice_dim + self.embed_dim + 1
        bases = IntegerMatrix(rows, cols)
        ## first n row
        for row in range(rows - 1):
            bases[row, row] = 2
            for col in range(self.embed_dim):
                bases[row, self.lattice_dim + 1 + col] = N * self.multiset[row][col]
        ## last row
        for col in range(cols):
            if col <= self.lattice_dim:
                bases[rows - 1, col] = 1
            else:
                bases[rows - 1, col] = N * self.target[col - self.lattice_dim - 1]
        # print(f"bases:\n{bases}")

        reduced_bases = LLL.reduction(bases)
        # print(f"reduced:\n{reduced_bases}")

        # there can be multiple solutions
        for row in reduced_bases:
            row = np.array(row)
            if self.is_solution(row):
                sol = np.abs(row[: self.lattice_dim] - row[self.lattice_dim]) // 2
                solutions.append(sol.astype(bool))

        return solutions

    def is_solution(self, row: np.ndarray):
        """
        given a row in the reduced LLL, decide if it's a potential subset sum solution
        """
        if (
            row[self.lattice_dim] == 1
            and np.all(row[self.lattice_dim + 1 :] == 0)
            and np.all(np.abs(row[: self.lattice_dim]) == 1)
        ):
            return True
        return False

    def solve_and_verify(self, expected: np.ndarray):
        """
        Solve the instance and check if the expected answer is among the found solutions
        Return two booleans, first indicates if the `expected` is found, second indicates
        an alternative subset is found
        """
        solutions = self.solve()
        found = np.any([np.all(expected == sol) for sol in solutions])
        forged = (
            np.any(
                [
                    (
                        np.count_nonzero(sol) == np.count_nonzero(expected)
                        and np.any(expected != sol)
                    )
                    for sol in solutions
                ]
            )
        )
        found_alt = not found and not forged and len(solutions) > 0
        # print(f"solutions: {solutions}")
        if found:
            print("Original solution found!")
        elif forged:
            print("Forgery found!")
        elif found_alt:
            print("Alternative (non-forgery) solution found!")
        else:
            print("Solution not found, consider a lower density for SSP.")
        return found, found_alt


class TestSubsetSum(unittest.TestCase):
    def test_success(self):
        np.random.seed(42)
        multiset = np.random.randint(-1000, 10000, size=10)
        subset_bit_vector = np.random.choice([0, 1], size=10, p=[0.8, 0.2]).astype(bool)
        target = np.sum(multiset * subset_bit_vector[:], axis=0)
        self.assertTrue(
            SubsetSum(multiset, target).solve_and_verify(
                subset_bit_vector.astype(bool)
            )[0]
        )

        n = 100
        multiset = np.random.randint(-(2**31), 2**31, size=(n, 50_000))
        p = 64 / n
        subset_bit_vector = np.random.choice([0, 1], size=n, p=[1 - p, p]).astype(bool)
        target = np.sum(
            multiset * subset_bit_vector[:, np.newaxis], axis=0
        )
        self.assertTrue(
            SubsetSum(multiset, target).solve_and_verify(
                subset_bit_vector.astype(bool)
            )[0]
        )


if __name__ == "__main__":
    unittest.main()
