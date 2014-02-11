import unittest
from means.approximation.ode_problem import Moment
import sympy

from means.approximation.mea.log_normal_closer import *

class TestLogNormalCloser(unittest.TestCase):

    def test_compute_raw_moments(self):
        """
        Given two vectors of Moments: counter and mcounter (up to second moment) and
        Given a vector of two species ymat,
        Then, the answer should match exactlty the expected result
        :return:
        """
        n_species = 3
        problem_moments = [
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

