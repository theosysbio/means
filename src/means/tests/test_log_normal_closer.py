import unittest
from means.approximation.ode_problem import Moment
import sympy

from means.approximation.mea.log_normal_closer import *

class TestLogNormalCloser(unittest.TestCase):

    __problem_moments = [
            Moment([0, 0, 0], symbol=sympy.Integer(1)),
            Moment([1, 0, 0], symbol=sympy.Symbol("y_0")),
            Moment([0, 1, 0], symbol=sympy.Symbol("y_1")),
            Moment([0, 0, 1], symbol=sympy.Symbol("y_2")),
            Moment([0, 0, 2], symbol=sympy.Symbol("yx2")),
            Moment([0, 1, 1], symbol=sympy.Symbol("yx3")),
            Moment([0, 2, 0], symbol=sympy.Symbol("yx4")),
            Moment([1, 0, 1], symbol=sympy.Symbol("yx5")),
            Moment([1, 1, 0], symbol=sympy.Symbol("yx6")),
            Moment([2, 0, 0], symbol=sympy.Symbol("yx7")),
            Moment([0, 0, 3], symbol=sympy.Symbol("yx8")),
            Moment([0, 1, 2], symbol=sympy.Symbol("yx9")),
            Moment([0, 2, 1], symbol=sympy.Symbol("yx10")),
            Moment([0, 3, 0], symbol=sympy.Symbol("yx11")),
            Moment([1, 0, 2], symbol=sympy.Symbol("yx12")),
            Moment([1, 1, 1], symbol=sympy.Symbol("yx13")),
            Moment([1, 2, 0], symbol=sympy.Symbol("yx14")),
            Moment([2, 0, 1], symbol=sympy.Symbol("yx15")),
            Moment([2, 1, 0], symbol=sympy.Symbol("yx16")),
            Moment([3, 0, 0], symbol=sympy.Symbol("yx17"))
        ]
    def test_get_log_covariance(self):


        log_variance_mat =sp.Matrix([
                ["log(1+yx7/y_0**2)", "0", "0"],
                ["0", "log(1+yx4/y_1**2)", "0"],
                ["0", "0", "log(1+yx2/y_2**2)"]
                        ])

        log_expectation_symbols = sp.Matrix([
                ["log(y_0)-log(1+yx7/y_0**2)/2"],
                ["log(y_1)-log(1+yx4/y_1**2)/2"],
                ["log(y_2)-log(1+yx2/y_2**2)/2"]
                ])

        covariance_matrix = sp.Matrix([
                ["yx7","yx6","yx5"],
                ["yx6","yx4","yx3"],
                ["yx5","yx3","yx2"]])

        expected = sp.sympify("log(1 + yx6/(y_0*y_1))")
        answer = get_log_covariance(log_variance_mat, log_expectation_symbols, covariance_matrix, 0,1)

        self.assertEqual(answer, expected)

        answer1 = get_log_covariance(log_variance_mat, log_expectation_symbols, covariance_matrix, 1,2)
        answer2 = get_log_covariance(log_variance_mat, log_expectation_symbols, covariance_matrix, 1,2)
        #logcovariance between species 1 and 2  ==  covarianc between sp. 2 and 1
        self.assertEqual(answer1, answer2)

    def test_get_covariance_symbol(self):

        problem_moments = self.__problem_moments
        expected = sp.Symbol("yx3")
        answer = get_covariance_symbol(problem_moments,1, 2)

        self.assertEqual(answer, expected)


        expected = sp.Symbol("yx6")
        answer = get_covariance_symbol(problem_moments,1, 0)

        self.assertEqual(answer, expected)

        #covariance between species 1 and 2  ==  covarianc between sp. 2 and 1
        answer1 = get_covariance_symbol(problem_moments,1, 0)
        answer2 = get_covariance_symbol(problem_moments,0, 1)
        self.assertEqual(answer1, answer2)


    def test_compute_raw_moments(self):
        """
        Given two vectors of Moments: counter and mcounter (up to second moment) and
        Given a vector of two species ymat,
        Then, the answer should match exactlty the expected result
        :return:
        """
        n_species = 3
        problem_moments = self.__problem_moments

        expected = sympy.Matrix([
            ["y_2**2+yx2"],
            ["y_1*y_2+yx3"],
            ["y_1**2+yx4"],
            ["y_0*y_2+yx5"],
            ["y_0*y_1+yx6"],
            ["y_0**2+yx7"],
            ["y_2**3+3*y_2*yx2+3*yx2**2/y_2+yx2**3/y_2**3"],
            ["y_1*y_2**2+y_1*yx2+2*y_2*yx3+2*yx2*yx3/y_2+yx3**2/y_1+yx2*yx3**2/(y_1*y_2**2)"],
            ["y_1**2*y_2+2*y_1*yx3+y_2*yx4+yx3**2/y_2+2*yx3*yx4/y_1+yx3**2*yx4/(y_1**2*y_2)"],
            ["y_1**3+3*y_1*yx4+3*yx4**2/y_1+yx4**3/y_1**3"],
            ["y_0*y_2**2+y_0*yx2+2*y_2*yx5+2*yx2*yx5/y_2+yx5**2/y_0+yx2*yx5**2/(y_0*y_2**2)"],
            ["y_0*y_1*y_2+y_0*yx3+y_1*yx5+y_2*yx6+yx3*yx5/y_2+yx3*yx6/y_1+yx5*yx6/y_0+yx3*yx5*yx6/(y_0*y_1*y_2)"],
            ["y_0*y_1**2+y_0*yx4+2*y_1*yx6+2*yx4*yx6/y_1+yx6**2/y_0+yx4*yx6**2/(y_0*y_1**2)"],
            ["y_0**2*y_2+2*y_0*yx5+y_2*yx7+yx5**2/y_2+2*yx5*yx7/y_0+yx5**2*yx7/(y_0**2*y_2)"],
            ["y_0**2*y_1+2*y_0*yx6+y_1*yx7+yx6**2/y_1+2*yx6*yx7/y_0+yx6**2*yx7/(y_0**2*y_1)"],
            ["y_0**3+3*y_0*yx7+3*yx7**2/y_0+yx7**3/y_0**3"]
        ])

        answer = compute_raw_moments(n_species,problem_moments)

        self.assertEqual(answer, expected)

