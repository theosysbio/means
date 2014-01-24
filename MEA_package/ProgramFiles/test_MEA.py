from MEA import make_damat
from MEA import substitute_mean_with_y
from MEA import substitute_raw_with_central
from MEA import substitute_ym_with_yx
from MEA import make_mfk

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

    def test_substitute_mean_with_y_simple_list(self):

        """
        Given the a list of expressions "mom" and
        Given the number of species is 2
        Then, "mom" should be substituted by "expected_mom"

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


        out_mom = substitute_mean_with_y(mom, nvar)

        self.assertEqual(Matrix(out_mom), Matrix(expected_mom))



    def test_substitute_mean_with_y_list_of_list(self):

        """
        Given the a list of lists of expressions "mom" and
        Given the number of species is 2
        Then, "mom" should be substituted by "expected_mom"

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


        out_mom = substitute_mean_with_y(mom, nvar)

        self.assertEqual(Matrix(out_mom), Matrix(expected_mom))


    def test_substitute_raw_with_central(self):

        """
        Given the a list of lists of central moment "central_moment",
        Given the symbols for the central moments "momvec", and
        Given the expressions of central moments in terms of raw moments "mom"
        Then, "central_moments" should be substituted by "expected_central_moments"

        :return:
        """

        central_moments = simplify([
            ["c_2*(-y_0 - y_1 + 301)", "-2*c_2", "-2*c_2", "0"],
            ["-c_0*y_0*y_1*(y_0 + y_1 - 181) + c_1*y_1*(-y_0 - y_1 + 301) - y_1*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301))", "-c_0*y_0 - c_1", "-c_0*y_0 - c_0*(y_0 + y_1 - 181) - c_1 - c_2", "-c_2"],
            ["-2*c_0*y_0**2*(y_0 + y_1 - 181) + c_0*y_0*(y_0 + y_1 - 181) + 2*c_1*y_0*(-y_0 - y_1 + 301) + c_1*(-y_0 - y_1 + 301) - 2*y_0*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301))", "0", "-2*c_0*y_0 + c_0 - 2*c_1", "-2*c_0*y_0 - 2*c_0*(y_0 + y_1 - 181) + c_0 - 2*c_1"]
        ])

        expected_central_moments = simplify([
            ["c_2*(-y_0 - y_1 + 301)", "-2*c_2", "-2*c_2", "0"],
            ["-c_0*y_0*y_1*(y_0 + y_1 - 181) + c_1*y_1*(-y_0 - y_1 + 301) - y_1*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301))", "-c_0*y_0 - c_1", "-c_0*y_0 - c_0*(y_0 + y_1 - 181) - c_1 - c_2", "-c_2"],
            ["-2*c_0*y_0**2*(y_0 + y_1 - 181) + c_0*y_0*(y_0 + y_1 - 181) + 2*c_1*y_0*(-y_0 - y_1 + 301) + c_1*(-y_0 - y_1 + 301) - 2*y_0*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301))", "0", "-2*c_0*y_0 + c_0 - 2*c_1", "-2*c_0*y_0 - 2*c_0*(y_0 + y_1 - 181) + c_0 - 2*c_1"]
        ])


        mom = simplify(["x02 - y_1**2", "x11 - y_0*y_1", "x20 - y_0**2"])


        momvec = simplify(["ym02", "ym11", "ym20"])

        central_moments = substitute_raw_with_central(central_moments, momvec, mom)

        self.assertEqual(Matrix(central_moments), Matrix(expected_central_moments))

    def test_substitute_ym_with_yx(self):
        """
        Given the a list of lists of central moment "central_moment", and
        Given the symbols for the central moments "momvec",
        Then, "central_moments" should be substituted by "expected_central_moments".

        :return:
        """

        momvec = simplify(["ym02", "ym11", "ym20", "ym03"])
        central_moments = simplify(
            [
                ["ym02 * 3", "ym11 + 32 + x", "ym20 + y_0"],
                ["ym01 * 3", "ym11 + 32 + x", "ym20 + y_0"],
                ["ym02 * 3", "ym11 + 32 + x", "ym03 + y_0"]
            ])


        expected_central_moments = simplify(
            [
                ["yx1 * 3", "yx2 + 32 + x", "yx3 + y_0"],
                ["ym01 * 3", "yx2 + 32 + x", "yx3 + y_0"],
                ["yx1 * 3", "yx2 + 32 + x", "yx4 + y_0"]
            ])

        central_moments = substitute_ym_with_yx(central_moments, momvec)

        self.assertEqual(Matrix(central_moments), Matrix(expected_central_moments))

    def test_make_mfk(self):

        """
        Given the Matrix "M",
        Given the vector of symbol for central moments "yms", and
        Given the "central_moments" matrix,
        Then, "mfk" should be exactly equal to "expected_mfk".

        :return:
        """


        M = Matrix(simplify([["-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301)", "0", "-c_0", "-c_0"],
                    [ "c_2*(-y_0 - y_1 + 301)", "0",    "0",    "0"]]))

        yms = Matrix(["1", "yx1", "yx2", "yx3"])

        central_moments = simplify(
            [
                ["c_2*(-y_0 - y_1 + 301)", "-2*c_2", "-2*c_2", "0"],
                ["0", "-c_0*y_0 - c_1", "-2*c_0*y_0 - c_0*y_1 + 181*c_0 - c_1 - c_2", "-c_2"],
                ["c_0*y_0**2 + c_0*y_0*y_1 - 181*c_0*y_0 - c_1*y_0 - c_1*y_1 + 301*c_1", "0", "-2*c_0*y_0 + c_0 - 2*c_1", "-4*c_0*y_0 - 2*c_0*y_1 + 363*c_0 - 2*c_1"]
            ])

        mfk = make_mfk(central_moments, yms, M)
        expected_mfk = simplify(
            ["-c_0*y_0*(y_0 + y_1 - 181) - c_0*yx2 - c_0*yx3 - c_1*(y_0 + y_1 - 301)",
             "c_2*(-y_0 - y_1 + 301)", "c_2*(-y_0 - y_1 - 2*yx1 - 2*yx2 + 301)",
             "-c_2*yx3 - yx1*(c_0*y_0 + c_1) - yx2*(2*c_0*y_0 + c_0*y_1 - 181*c_0 + c_1 + c_2)",
             "c_0*y_0**2 + c_0*y_0*y_1 - 181*c_0*y_0 - c_1*y_0 - c_1*y_1 + 301*c_1 - yx2*(2*c_0*y_0 - c_0 + 2*c_1) - yx3*(4*c_0*y_0 + 2*c_0*y_1 - 363*c_0 + 2*c_1)"]
        )


        self.assertEqual(Matrix(mfk), Matrix(expected_mfk))