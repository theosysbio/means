from MEA import make_damat
import unittest
from sympy import Matrix, diff, Symbol, Subs, Eq, var, simplify, S
import re


class MEATestCase(unittest.TestCase):
    def test_make_damat(self):
        """
        Given the number of moments is 3, the number of species is 2,
        And Given the propensities of the 3 reactions in `a_strings`,
        Then results of make_damat() should produce the derivation "matrix" (`result_damat`)
        exactly equal to the the expected one (`expected_damat`).

        :return:
        """

        nMoments, nvariables = 3, 2

        ymat = Matrix(nvariables, 1, lambda i, j : 0)
        for i in range(nvariables):
            ymat[i,0] = Symbol("y_%i" % i)

        a_strings = ["c_0*y_0*(120-301+y_0+y_1)", "c_1*(301-(y_0+y_1))", "c_2*(301-(y_0+y_1))"]

        nreactions = len(a_strings)

        amat = Matrix(nMoments, 1, lambda i, j : simplify(a_strings[i]))

        result_damat = make_damat(amat, nMoments, ymat)

        expected_damat = Matrix(nMoments, 1, lambda i, j : 0)

        z = S(0)
        expected_damat[0,0] = [[diff(simplify(a_strings[0]), Symbol("y_0")), simplify("c_0*y_0")],
                                [simplify("-c_1"), simplify("-c_1")],
                                [simplify("-c_2"), simplify("-c_2")]]

        expected_damat[1,0] = [[simplify("2*c_0"), simplify("c_0"), simplify("c_0"), z],
                                [z] * 4,
                                [z] * 4]

        expected_damat[2,0] = [[z]*8] * 3

        self.assertEqual(expected_damat, result_damat)
