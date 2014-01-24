from MEA import make_damat
from MEA import substitute_mean_with_y
from MEA import substitute_raw_with_central
from MEA import substitute_ym_with_yx

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
        print "boooooooo1"
        self.assertEqual(expected_damat, result_damat)

    def test_substitute_mean_with_y_simple_list(self):

        """
        Given the number of moments is 3, the number of species is 2,
        And Given the propensities of the 3 reactions in `a_strings`,
        Then results of make_damat() should produce the derivation "matrix" (`result_damat`)
        exactly equal to the the expected one (`expected_damat`).

        :return:
        """

        nvar = 2
        mom = simplify([
        "-2*x01*y_1 + x02 + y_1**2",
        " -x01*y_0 - x10*y_1 + x11 + y_0*y_1",
        " -2*x10*y_0 + x20 + y_0**2"])

        expected_mom = simplify([
        "-2*y_1*y_1 + x02 + y_1**2",
        " -y_1*y_0 - y_0*y_1 + x11 + y_0*y_1",
        "-2*y_0*y_0 + x20 + y_0**2"])


        out_mom =  substitute_mean_with_y(mom, nvar)

        self.assertEqual(Matrix(out_mom), Matrix(expected_mom))



    def test_substitute_mean_with_y_list_of_list(self):

        """
        Given the number of moments is 3, the number of species is 2,
        And Given the propensities of the 3 reactions in `a_strings`,
        Then results of make_damat() should produce the derivation "matrix" (`result_damat`)
        exactly equal to the the expected one (`expected_damat`).

        :return:
        """

        nvar = 2
        mom = simplify([
            ["-2*c_2*x01*(-y_0 - y_1 + 301)", "-2*c_2"],
            ["c_2*x10*(-y_0 - y_1 + 301)- x01", "x01 - c_0*y_0 - c_0*y_1"],
            ["0","2 * x10 * (-c_0 * y_0 * (y_0 + y_1 - 181))"]
        ])


        expected_mom = simplify([
            ["-2*c_2*y_1*(-y_0 - y_1 + 301)", "-2*c_2"],
            ["c_2*y_0*(-y_0 - y_1 + 301)- y_1", "y_1 - c_0*y_0 - c_0*y_1"],
            ["0", "2*y_0*(-c_0*y_0*(y_0 + y_1 - 181))"]
        ])


        out_mom =  substitute_mean_with_y(mom, nvar)

        self.assertEqual(Matrix(out_mom), Matrix(expected_mom))





