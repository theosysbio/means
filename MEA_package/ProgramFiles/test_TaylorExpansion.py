from TaylorExpansion import TaylorExpansion
from fcount import fcount
import unittest
from sympy import Matrix, diff, Symbol, Subs, Eq, var, simplify, S

class TaylorExpansionTestCase(unittest.TestCase):
    def test_TaylorExpansion(self):
        """
        Given the number of moments is 3, the number of species is 2,
        Given the propensities of the 3 reactions in `a_strings`,
        And Given the combination of derivative order in counter,
        Then results of `TaylorExpansion()` should produce a matrix exactly equal to
        exactly equal to the the expected one (`expected_te_matrix`).

        :return:
        """


        nMoments = 3
        nvariables = 2
        a_strings = ["c_0*y_0*(120-301+y_0+y_1)", "c_1*(301-(y_0+y_1))", "c_2*(301-(y_0+y_1))"]
        z = S(0)

        damat = Matrix(nMoments, 1, lambda i, j : 0)

        damat[0,0] = [[diff(simplify(a_strings[0]), Symbol("y_0")), simplify("c_0*y_0")],
                                [simplify("-c_1"), simplify("-c_1")],
                                [simplify("-c_2"), simplify("-c_2")]]

        damat[1,0] = [[simplify("2*c_0"), simplify("c_0"), simplify("c_0"), z],
                                [z] * 4,
                                [z] * 4]
        damat[2,0] = [[z]*8] * 3

        nreactions = len(a_strings)

        amat = Matrix(nMoments, 1, lambda i, j : simplify(a_strings[i]))

        counter = fcount(nMoments, nvariables)[0]

        te_result = TaylorExpansion(nreactions, nvariables, damat, amat, counter, nMoments)

        # hardcodding the expected matrix:
        expected_te_mat = Matrix(nreactions, len(counter), lambda i, j : 0)

        expected_te_mat[0,0] = simplify("c_0*y_0*(y_0 + y_1 - 181)")
        expected_te_mat[1,0] = simplify("c_1*(-y_0 - y_1 + 301)")
        expected_te_mat[2,0] = simplify("c_2*(-y_0 - y_1 + 301)")
        expected_te_mat[0, 2],expected_te_mat[0, 3] = (Symbol("c_0"),Symbol("c_0"))

        self.assertEqual(expected_te_mat, te_result)
