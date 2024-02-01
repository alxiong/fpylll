# Module for solving Subset Sum Problem using LLL

import unittest
import numpy as np
from fpylll import LLL, IntegerMatrix


class SubsetSum():
    """
    Class for defining and solving Subset Sum problem
    """
    def __init__(self, multiset: np.ndarray, target: np.ndarray, modulus: np.uint32):
        """
        Define the problem instance with {a_i}, T and q
        such that a_i, T in FF_q or FF_q^n for i in [1, m]
        """
        assert multiset.dtype == np.uint32 and target.dtype == np.uint32
        assert multiset.ndim == target.ndim + 1
        assert multiset.ndim in [1, 2]

        self.multiset = multiset
        self.target = target
        self.modulus = modulus
        # lattice dim: n
        self.dim = 1 if target.ndim == 0 else target.shape[0]
        # number of instances: m
        self.instance_size = multiset.shape[0]
        print(f"n: {self.dim}, m: {self.instance_size}")

    def solve(self):
        """
        Solve the instance, return an array of value: `z in {0, 1}^len(self.multiset)`
        Such that `A*z = t mod q`
        """

        # construct the bases matrix B which is (m+n+1) by (m+n+1) matrix:
        # where .i represents the i-th dimension (i in [n])
        # [ 1 0 .. 0 0      a_1.1   a_1.2 ... a_1.n ]
        # [ 0 1 .. 0 0      a_2.1   a_2.2 ... a_2.n ]
        # [                                    ]
        # [ 0 0 .. 1 0      a_m.1   a_m.2 ... a_m.n ]
        # [ 0 0 .. 0 1      -t.1    -t.2  ... -t.n  ]
        # [ 0 0 .. 0 0      q       0     ... 0     ]
        # [ 0 0 .. 0 0      0       q     ... 0     ]
        # [                                         ]
        # [ 0 0 .. 0 0      0       0     ... q     ]
        bases = IntegerMatrix(self.instance_size + self.dim + 1, self.instance_size + self.dim + 1)
        # diagonal value
        for i in range(self.instance_size + self.dim + 1):
            if i <= self.instance_size:
                bases[i, i] = 1
            else:
                bases[i, i] = self.modulus

        # top right
        for i in range(self.instance_size):
            if self.dim == 1:
                bases[i, -1] = self.multiset[i]
            else:
                for j in range(self.dim):
                    bases[i, self.instance_size + j] = self.multiset[i, j]
        # middle row: target value
        if self.dim == 1:
            bases[self.instance_size, self.instance_size + 1] = self.modulus - self.target
        else:
            for j in range(self.dim):
                bases[self.instance_size, self.instance_size + j] = self.modulus - self.target[j]
        # print(f"bases:\n{bases}")

        reduced_bases = LLL.reduction(bases)
        # print(f"reduced:\n{reduced_bases}")

        # there can be multiple solutions
        solutions = []
        for row in reduced_bases:
            row = np.array(row)
            found, sol = self.is_solution(row)
            if found:
                solutions.append(sol.tolist())
        return solutions

    def is_solution(self, row: np.ndarray):
        """
        given a row in the reduced LLL, decide if it's a potential subset sum solution
        give (bool, solution)
        """
        unique_elems = set(np.unique(row[:self.instance_size]))
        if len(unique_elems) == 2 and 0 in unique_elems:
            if np.all(row[self.instance_size:]) % self.modulus == 0:
                candidate_solution = row[:self.instance_size]
                quotient = (unique_elems - {0}).pop()
                return True, candidate_solution // quotient
        return False, None

    # TODO: WIP: this is not working yet
    def solve_ahl(self):
        """
        Using [AHL] method: https://link.springer.com/chapter/10.1007/3-540-69346-7_18
        """
        # N1, N2 can be any positive integers
        N1 = 100
        N2 = 1000

        bases = IntegerMatrix(self.instance_size + 1, self.instance_size + self.dim + 1)
        for i in range(self.instance_size):
            bases[i, i] = 1

            if self.dim == 1:
                bases[i, -1] = self.multiset[i]
            else:
                for j in range(self.instance_size + 1, self.instance_size + self.dim + 1):
                    bases[i, j] = N2 * self.multiset[i, j - self.instance_size - 1]
        if self.dim == 1:
            bases[-1, -2] = N1
            bases[-1, -1] = self.modulus - ((N2 * self.target) % self.modulus)
        else:
            for j in range(self.dim):
                bases[-1, self.instance_size + j + 1] = self.modulus - ((N2 * self.target[j]) % self.modulus)
        print(f"bases:\n{bases}")

        reduced_bases = LLL.reduction(bases)

        print(f"reduced:\n{reduced_bases}")
        for row in reduced_bases:
            row = np.array(row)
            if row[self.instance_size] == N1 and np.all(row[self.instance_size + 1:] % self.modulus == 0):
                return row[:self.instance_size].tolist()
        return None


class TestSubsetSum(unittest.TestCase):
    def test_success(self):
        solution = SubsetSum(
            np.array([101, 221, 325, 124, 552, 612, 737], dtype=np.uint32),
            np.array(838, dtype=np.uint32),
            1024
        ).solve()
        self.assertTrue([1, 0, 0, 0, 0, 0, 1] in solution)

        solution = SubsetSum(
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.uint32),
            np.array([6, 1], dtype=np.uint32),
            7
        ).solve()
        self.assertTrue([1, 0, 1] in solution)

if __name__ == '__main__':
    unittest.main()
