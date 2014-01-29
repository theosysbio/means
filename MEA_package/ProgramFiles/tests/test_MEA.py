#from MEA import make_damat
from MEA import substitute_mean_with_y
from MEA import substitute_raw_with_central
from MEA import substitute_ym_with_yx
from MEA import make_mfk

import unittest
from sympy import Matrix, diff, Symbol, Subs, Eq, var, simplify, S
import re


class MEATestCase(unittest.TestCase):

    def test_substitute_mean_with_y_simple_list(self):

        """
        Given the a list of expressions "mom" and
        Given the number of species is 2
        Then, "mom" should be substituted by "expected_mom"

        :return:
        """

        nvar = 2
        mom = simplify([
        "-2*x_0_1*y_1 + x_0_2 + y_1**2",
        " -x_0_1*y_0 - x_1_0*y_1 + x_1_1 + y_0*y_1",
        " -2*x_1_0*y_0 + x_2_0 + y_0**2"])

        expected_mom = simplify([
        "-2*y_1*y_1 + x_0_2 + y_1**2",
        " -y_1*y_0 - y_0*y_1 + x_1_1 + y_0*y_1",
        "-2*y_0*y_0 + x_2_0 + y_0**2"])



        # print mom
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
            ["-2*c_2*x_0_1*(-y_0 - y_1 + 301)", "-2*c_2"],
            ["c_2*x_1_0*(-y_0 - y_1 + 301)- x_0_1", "x_0_1 - c_0*y_0 - c_0*y_1"],
            ["0","2 * x_1_0 * (-c_0 * y_0 * (y_0 + y_1 - 181))"]
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


        mom = simplify(["x_0_2 - y_1**2", "x_1_1 - y_0*y_1", "x_2_0 - y_0**2"])


        momvec = simplify(["ym_0_2", "ym_1_1", "ym_2_0"])

        central_moments = substitute_raw_with_central(central_moments, momvec, mom)

        self.assertEqual(Matrix(central_moments), Matrix(expected_central_moments))

    def test_substitute_ym_with_yx(self):
        """
        Given the a list of lists of central moment "central_moment", and
        Given the symbols for the central moments "momvec",
        Then, "central_moments" should be substituted by "expected_central_moments".

        :return:
        """

        momvec = simplify(["ym_0_2", "ym_1_1", "ym_2_0", "ym_0_3"])
        central_moments = simplify(
            [
                ["ym_0_2 * 3", "ym_1_1 + 32 + x", "ym_2_0 + y_0"],
                ["ym_0_1 * 3", "ym_1_1 + 32 + x", "ym_2_0 + y_0"],
                ["ym_0_2 * 3", "ym_1_1 + 32 + x", "ym_0_3 + y_0"]
            ])


        expected_central_moments = simplify(
            [
                ["yx1 * 3", "yx2 + 32 + x", "yx3 + y_0"],
                ["ym_0_1 * 3", "yx2 + 32 + x", "yx3 + y_0"],
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