import unittest
import sympy
from means.approximation.mea.gamma_closer import GammaCloser
from means.approximation.ode_problem import Moment
from means.util.sympyhelpers import sympy_expressions_equal
from means.util.sympyhelpers import to_sympy_matrix



import unittest
from means.approximation.ode_problem import Moment
import sympy
from means.approximation.mea.log_normal_closer import LogNormalCloser
from means.util.sympyhelpers import sympy_expressions_equal
from means.util.sympyhelpers import to_sympy_matrix

## for univariate or multivariate on dimerisation Model, we should have:
# m = to_sympy_matrix([
#             ["c_1*(c_2-y_0)-2*c_0*yx2-2*c_0*y_0*(y_0-1)"],
#             ["2*c_1*c_2-4*c_0*y_0-2*c_1*y_0+8*c_0*yx2-4*c_0*yx3-2*c_1*yx2+4*c_0*y_0 ** 2-8*c_0*y_0*yx2"],
#             ["4*c_1*c_2+8*c_0*y_0-4*c_1*y_0-6*c_0*ym4-20*c_0*yx2+18*c_0*yx3-6*c_1*yx2-3*c_1*yx3-8*c_0*y_0 ** 2+6*c_0*yx2 ** 2+24*c_0*y_0*yx2-12*c_0*y_0*yx3"]
# ])


class TestLogNormalCloser(unittest.TestCase):
    __n_counter = [
    Moment([0, 0, 0], symbol=sympy.Integer(0)),
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
    Moment([3, 0, 0], symbol=sympy.Symbol("yx17")),

    ]
    __k_counter = [
        Moment([0, 0, 0], symbol=sympy.Integer(1)),
        Moment([1, 0, 0], symbol=sympy.Symbol("y_0")),
        Moment([0, 1, 0], symbol=sympy.Symbol("y_1")),
        Moment([0, 0, 1], symbol=sympy.Symbol("y_2")),
        Moment([0, 0, 2], symbol=sympy.Symbol("x_0_0_2")),
        Moment([0, 1, 1], symbol=sympy.Symbol("x_0_1_1")),
        Moment([0, 2, 0], symbol=sympy.Symbol("x_0_2_0")),
        Moment([1, 0, 1], symbol=sympy.Symbol("x_1_0_1")),
        Moment([1, 1, 0], symbol=sympy.Symbol("x_1_1_0")),
        Moment([2, 0, 0], symbol=sympy.Symbol("x_2_0_0")),
        Moment([0, 0, 3], symbol=sympy.Symbol("x_0_0_3")),
        Moment([0, 1, 2], symbol=sympy.Symbol("x_0_1_2")),
        Moment([0, 2, 1], symbol=sympy.Symbol("x_0_2_1")),
        Moment([0, 3, 0], symbol=sympy.Symbol("x_0_3_0")),
        Moment([1, 0, 2], symbol=sympy.Symbol("x_1_0_2")),
        Moment([1, 1, 1], symbol=sympy.Symbol("x_1_1_1")),
        Moment([1, 2, 0], symbol=sympy.Symbol("x_1_2_0")),
        Moment([2, 0, 1], symbol=sympy.Symbol("x_2_0_1")),
        Moment([2, 1, 0], symbol=sympy.Symbol("x_2_1_0")),
        Moment([3, 0, 0], symbol=sympy.Symbol("x_3_0_0"))
    ]
    __mfk = to_sympy_matrix([
    ["c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0) - c_2*y_2*yx17*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - c_2*y_2*yx7*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*yx15*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*yx5*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)"],
    ["c_3*y_0 - c_4*y_1"],
    ["c_4*y_1 - c_5*y_2"],
    ["2*c_4*y_1*y_2 + c_4*y_1 + 2*c_4*yx3 - 2*c_5*y_2**2 + c_5*y_2 - 2*c_5*yx2 - 2*y_2*(c_4*y_1 - c_5*y_2)"],
    ["c_3*y_0*y_2 + c_3*yx5 + c_4*y_1**2 - c_4*y_1*y_2 - c_4*y_1 + c_4*yx4 - c_5*y_1*y_2 - y_1*(c_4*y_1 - c_5*y_2) - y_2*(c_3*y_0 - c_4*y_1) + yx3*(-c_4 - c_5)"],
    ["2*c_3*y_0*y_1 + c_3*y_0 + 2*c_3*yx6 - 2*c_4*y_1**2 + c_4*y_1 - 2*c_4*yx4 - 2*y_1*(c_3*y_0 - c_4*y_1)"],
    ["c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) - c_2*y_0*yx2/(c_6 + y_0) - c_2*y_2*yx15*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*yx12*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_4*y_0*y_1 + c_4*yx6 - c_5*y_0*y_2 - y_0*(c_4*y_1 - c_5*y_2) - y_2*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + yx5*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_5)"],
    ["c_0*y_1 - c_1*y_0*y_1 - c_2*y_0*y_1*y_2/(c_6 + y_0) - c_2*y_0*yx3/(c_6 + y_0) - c_2*y_2*yx16*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - c_2*yx13*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_3*y_0**2 + c_3*yx7 - c_4*y_0*y_1 - y_0*(c_3*y_0 - c_4*y_1) - y_1*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + yx6*(-c_1 - c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - c_4)"],
    ["2*c_0*y_0 + c_0 - 2*c_1*y_0**2 + c_1*y_0 - 2*c_2*y_0**2*y_2/(c_6 + y_0) + c_2*y_0*y_2/(c_6 + y_0) - 2*y_0*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) + yx15*(2*c_2*y_0*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2) + yx17*(2*c_2*y_0*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3 - 2*c_2*y_2*(-y_0**2/(c_6 + y_0)**2 + 2*y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 + c_2*y_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)**3) + yx5*(2*c_2*y_0*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0) - 2*c_2*y_0*(-y_0/(c_6 + y_0) + 2)/(c_6 + y_0) + c_2*(-y_0/(c_6 + y_0) + 1)/(c_6 + y_0)) + yx7*(-2*c_1 + 2*c_2*y_0*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2 - 2*c_2*y_2*(y_0**2/(c_6 + y_0)**2 - 2*y_0/(c_6 + y_0) + 1)/(c_6 + y_0) + c_2*y_2*(y_0/(c_6 + y_0) - 1)/(c_6 + y_0)**2)"]
    ])

    def test_close_type_one(self):
        central_from_raw_exprs = to_sympy_matrix(
                    [["x_0_0_2-y_2**2"],
                    ["x_0_1_1-y_1*y_2"],
                    ["x_0_2_0-y_1**2"],
                    ["x_1_0_1-y_0*y_2"],
                    ["x_1_1_0-y_0*y_1"],
                    ["x_2_0_0-y_0**2"],
                    ["-3*x_0_0_2*y_2+x_0_0_3+2*y_2**3"],
                    ["-x_0_0_2*y_1-2*x_0_1_1*y_2+x_0_1_2+2*y_1*y_2**2"],
                    ["-2*x_0_1_1*y_1-x_0_2_0*y_2+x_0_2_1+2*y_1**2*y_2"],
                    ["-3*x_0_2_0*y_1+x_0_3_0+2*y_1**3"],
                    ["-x_0_0_2*y_0-2*x_1_0_1*y_2+x_1_0_2+2*y_0*y_2**2"],
                    ["-x_0_1_1*y_0-x_1_0_1*y_1-x_1_1_0*y_2+x_1_1_1+2*y_0*y_1*y_2"],
                    ["-x_0_2_0*y_0-2*x_1_1_0*y_1+x_1_2_0+2*y_0*y_1**2"],
                    ["-2*x_1_0_1*y_0-x_2_0_0*y_2+x_2_0_1+2*y_0**2*y_2"],
                    ["-2*x_1_1_0*y_0-x_2_0_0*y_1+x_2_1_0+2*y_0**2*y_1"],
                    ["-3*x_2_0_0*y_0+x_3_0_0+2*y_0**3"]
         ])


        max_order = 2

        expected = to_sympy_matrix([
            ["c_0-c_1*y_0-(c_2*c_6*yx5)/(c_6+y_0) ** 2-(c_2*y_0*y_2)/(c_6+y_0)+(c_2*c_6*y_2*yx7)/(c_6+y_0) ** 3-(2*c_2*c_6*y_2*yx7 ** 2)/(y_0*(c_6+y_0) ** 4)+(2*c_2*c_6*yx2*yx7 ** 2)/(y_0 ** 2*y_2*(c_6+y_0) ** 3)"],
            ["c_3*y_0-c_4*y_1"],
            ["c_4*y_1-c_5*y_2"],
            ["c_4*y_1+c_5*y_2+2*c_4*yx3-2*c_5*yx2"],
            ["c_3*yx5-c_4*yx3-c_4*y_1+c_4*yx4-c_5*yx3"],
            ["c_3*y_0+c_4*y_1-2*c_4*yx4+2*c_3*yx6"],
            ["-(c_2*y_0 ** 5*y_2 ** 2*yx2+c_1*y_0 ** 5*y_2 ** 2*yx5-c_4*y_0 ** 5*y_2 ** 2*yx6+c_5*y_0 ** 5*y_2 ** 2*yx5+2*c_2*c_6*y_0 ** 4*y_2 ** 2*yx2+3*c_1*c_6*y_0 ** 4*y_2 ** 2*yx5+c_2*c_6*y_0 ** 3*y_2 ** 3*yx5-3*c_4*c_6*y_0 ** 4*y_2 ** 2*yx6+3*c_5*c_6*y_0 ** 4*y_2 ** 2*yx5+2*c_2*c_6*y_0 ** 2*yx2 ** 2*yx7+2*c_2*c_6 ** 2*y_0*yx2 ** 2*yx7-2*c_2*c_6*y_2 ** 2*yx2*yx7 ** 2+c_2*c_6 ** 2*y_0 ** 3*y_2 ** 2*yx2+3*c_1*c_6 ** 2*y_0 ** 3*y_2 ** 2*yx5+c_1*c_6 ** 3*y_0 ** 2*y_2 ** 2*yx5+c_2*c_6 ** 2*y_0 ** 2*y_2 ** 3*yx5-3*c_4*c_6 ** 2*y_0 ** 3*y_2 ** 2*yx6-c_4*c_6 ** 3*y_0 ** 2*y_2 ** 2*yx6+3*c_5*c_6 ** 2*y_0 ** 3*y_2 ** 2*yx5+c_5*c_6 ** 3*y_0 ** 2*y_2 ** 2*yx5)/(y_0 ** 2*y_2 ** 2*(c_6+y_0) ** 3)"],
            ["-(c_2*y_0 ** 5*y_1*y_2*yx3+c_1*y_0 ** 5*y_1*y_2*yx6-c_3*y_0 ** 5*y_1*y_2*yx7+c_4*y_0 ** 5*y_1*y_2*yx6-2*c_2*c_6*y_2 ** 2*yx4*yx7 ** 2+c_2*c_6 ** 2*y_0 ** 2*y_1*y_2 ** 2*yx6+2*c_2*c_6*y_0 ** 4*y_1*y_2*yx3+3*c_1*c_6*y_0 ** 4*y_1*y_2*yx6-3*c_3*c_6*y_0 ** 4*y_1*y_2*yx7+3*c_4*c_6*y_0 ** 4*y_1*y_2*yx6+2*c_2*c_6*y_0 ** 2*yx2*yx4*yx7+2*c_2*c_6 ** 2*y_0*yx2*yx4*yx7+c_2*c_6 ** 2*y_0 ** 3*y_1*y_2*yx3+3*c_1*c_6 ** 2*y_0 ** 3*y_1*y_2*yx6+c_1*c_6 ** 3*y_0 ** 2*y_1*y_2*yx6+c_2*c_6*y_0 ** 3*y_1*y_2 ** 2*yx6-3*c_3*c_6 ** 2*y_0 ** 3*y_1*y_2*yx7-c_3*c_6 ** 3*y_0 ** 2*y_1*y_2*yx7+3*c_4*c_6 ** 2*y_0 ** 3*y_1*y_2*yx6+c_4*c_6 ** 3*y_0 ** 2*y_1*y_2*yx6)/(y_0 ** 2*y_1*y_2*(c_6+y_0) ** 3)"],
            ["(c_0*y_0 ** 6*y_2+c_1*y_0 ** 7*y_2+c_2*y_0 ** 6*y_2 ** 2+3*c_2*c_6 ** 2*y_0 ** 4*y_2 ** 2+c_2*c_6 ** 3*y_0 ** 3*y_2 ** 2+4*c_0*c_6*y_0 ** 5*y_2+4*c_1*c_6*y_0 ** 6*y_2-2*c_2*y_0 ** 6*y_2*yx5-2*c_1*y_0 ** 6*y_2*yx7+6*c_0*c_6 ** 2*y_0 ** 4*y_2+4*c_0*c_6 ** 3*y_0 ** 3*y_2+c_0*c_6 ** 4*y_0 ** 2*y_2+6*c_1*c_6 ** 2*y_0 ** 5*y_2+4*c_1*c_6 ** 3*y_0 ** 4*y_2+c_1*c_6 ** 4*y_0 ** 3*y_2+3*c_2*c_6*y_0 ** 5*y_2 ** 2-2*c_2*c_6 ** 2*yx2*yx7 ** 2-4*c_2*c_6 ** 3*yx2*yx7 ** 2+2*c_2*c_6 ** 2*y_0 ** 3*y_2*yx5+c_2*c_6 ** 3*y_0 ** 2*y_2*yx5+2*c_2*c_6*y_0*y_2 ** 2*yx7 ** 2-6*c_2*c_6 ** 2*y_0 ** 4*y_2*yx5-2*c_2*c_6 ** 3*y_0 ** 3*y_2*yx5-12*c_1*c_6 ** 2*y_0 ** 4*y_2*yx7-8*c_1*c_6 ** 3*y_0 ** 3*y_2*yx7-2*c_1*c_6 ** 4*y_0 ** 2*y_2*yx7-c_2*c_6*y_0 ** 3*y_2 ** 2*yx7-2*c_2*c_6*y_0 ** 4*y_2 ** 2*yx7-4*c_2*c_6*y_0 ** 2*yx2*yx7 ** 2-8*c_2*c_6 ** 2*y_0*yx2*yx7 ** 2+4*c_2*c_6*y_0 ** 2*y_2 ** 2*yx7 ** 2+4*c_2*c_6 ** 2*y_0*y_2 ** 2*yx7 ** 2-c_2*c_6 ** 2*y_0 ** 2*y_2 ** 2*yx7-4*c_2*c_6 ** 2*y_0 ** 3*y_2 ** 2*yx7-2*c_2*c_6 ** 3*y_0 ** 2*y_2 ** 2*yx7+c_2*c_6*y_0 ** 4*y_2*yx5-6*c_2*c_6*y_0 ** 5*y_2*yx5-8*c_1*c_6*y_0 ** 5*y_2*yx7-2*c_2*c_6*y_0*yx2*yx7 ** 2)/(y_0 ** 2*y_2*(c_6+y_0) ** 4)"]
        ])

        closer = GammaCloser(max_order, type=1)
        answer = closer.close(self.__mfk, central_from_raw_exprs, self.__n_counter, self.__k_counter)
        self.assertTrue(sympy_expressions_equal(answer, expected))


    def test_close_type_zero(self):

        central_from_raw_exprs = to_sympy_matrix(
                    [["x_0_0_2-y_2**2"],
                    ["x_0_1_1-y_1*y_2"],
                    ["x_0_2_0-y_1**2"],
                    ["x_1_0_1-y_0*y_2"],
                    ["x_1_1_0-y_0*y_1"],
                    ["x_2_0_0-y_0**2"],
                    ["-3*x_0_0_2*y_2+x_0_0_3+2*y_2**3"],
                    ["-x_0_0_2*y_1-2*x_0_1_1*y_2+x_0_1_2+2*y_1*y_2**2"],
                    ["-2*x_0_1_1*y_1-x_0_2_0*y_2+x_0_2_1+2*y_1**2*y_2"],
                    ["-3*x_0_2_0*y_1+x_0_3_0+2*y_1**3"],
                    ["-x_0_0_2*y_0-2*x_1_0_1*y_2+x_1_0_2+2*y_0*y_2**2"],
                    ["-x_0_1_1*y_0-x_1_0_1*y_1-x_1_1_0*y_2+x_1_1_1+2*y_0*y_1*y_2"],
                    ["-x_0_2_0*y_0-2*x_1_1_0*y_1+x_1_2_0+2*y_0*y_1**2"],
                    ["-2*x_1_0_1*y_0-x_2_0_0*y_2+x_2_0_1+2*y_0**2*y_2"],
                    ["-2*x_1_1_0*y_0-x_2_0_0*y_1+x_2_1_0+2*y_0**2*y_1"],
                    ["-3*x_2_0_0*y_0+x_3_0_0+2*y_0**3"]
         ])

        max_order = 2
        expected = to_sympy_matrix([
            ["c_0-c_1*y_0-(c_2*c_6*yx5)/(c_6+y_0) ** 2-(c_2*y_0*y_2)/(c_6+y_0)+(c_2*c_6*y_2*yx7)/(c_6+y_0) ** 3-(2*c_2*c_6*y_2*yx7 ** 2)/(y_0*(c_6+y_0) ** 4)"],
            ["c_3*y_0-c_4*y_1"],
            ["c_4*y_1-c_5*y_2"],
            ["c_4*y_1+c_5*y_2+2*c_4*yx3-2*c_5*yx2"],
            ["c_3*yx5-c_4*yx3-c_4*y_1+c_4*yx4-c_5*yx3"],
            ["c_3*y_0+c_4*y_1-2*c_4*yx4+2*c_3*yx6"],
            ["c_4*yx6-c_1*yx5-c_5*yx5-(c_2*y_0*yx2)/(c_6+y_0)-(c_2*y_2*yx5)/(c_6+y_0)+(c_2*y_0*y_2*yx5)/(c_6+y_0) ** 2"],
            ["c_3*yx7-c_1*yx6-c_4*yx6-(c_2*y_0*yx3)/(c_6+y_0)-(c_2*y_2*yx6)/(c_6+y_0)+(c_2*y_0*y_2*yx6)/(c_6+y_0) ** 2"],
            ["(c_0*y_0 ** 5+c_1*y_0 ** 6+c_2*y_0 ** 5*y_2-2*c_2*y_0 ** 5*yx5-2*c_1*y_0 ** 5*yx7+6*c_0*c_6 ** 2*y_0 ** 3+4*c_0*c_6 ** 3*y_0 ** 2+6*c_1*c_6 ** 2*y_0 ** 4+4*c_1*c_6 ** 3*y_0 ** 3+c_1*c_6 ** 4*y_0 ** 2+4*c_0*c_6*y_0 ** 4+c_0*c_6 ** 4*y_0+4*c_1*c_6*y_0 ** 5+3*c_2*c_6*y_0 ** 4*y_2+c_2*c_6*y_0 ** 3*yx5+c_2*c_6 ** 3*y_0*yx5-6*c_2*c_6*y_0 ** 4*yx5-8*c_1*c_6*y_0 ** 4*yx7-2*c_1*c_6 ** 4*y_0*yx7+2*c_2*c_6*y_2*yx7 ** 2+3*c_2*c_6 ** 2*y_0 ** 3*y_2+c_2*c_6 ** 3*y_0 ** 2*y_2+2*c_2*c_6 ** 2*y_0 ** 2*yx5-6*c_2*c_6 ** 2*y_0 ** 3*yx5-2*c_2*c_6 ** 3*y_0 ** 2*yx5-12*c_1*c_6 ** 2*y_0 ** 3*yx7-8*c_1*c_6 ** 3*y_0 ** 2*yx7+4*c_2*c_6 ** 2*y_2*yx7 ** 2-4*c_2*c_6 ** 2*y_0 ** 2*y_2*yx7+4*c_2*c_6*y_0*y_2*yx7 ** 2-c_2*c_6*y_0 ** 2*y_2*yx7-c_2*c_6 ** 2*y_0*y_2*yx7-2*c_2*c_6*y_0 ** 3*y_2*yx7-2*c_2*c_6 ** 3*y_0*y_2*yx7)/(y_0*(c_6+y_0) ** 4)"]
        ])
        closer = GammaCloser(max_order, type=0)
        answer = closer.close(self.__mfk, central_from_raw_exprs, self.__n_counter, self.__k_counter)
        print (answer - expected).applyfunc(sympy.simplify)
        self.assertTrue(sympy_expressions_equal(answer, expected))

    #
    # def test_compute_closed_central_moments(self):
    #     expected = to_sympy_matrix([
    #         ["yx2"],
    #         ["(yx2*yx4)/(y_1*y_2)"],
    #         ["yx4"],
    #         ["(yx2*yx7)/(y_0*y_2)"],
    #         ["(yx4*yx7)/(y_0*y_1)"],
    #         ["yx7"],
    #         ["(2*yx2 ** 2)/y_2"],
    #         ["(2*yx2 ** 2*yx4)/(y_1*y_2 ** 2)"],
    #         ["(2*yx2*yx4 ** 2)/(y_1 ** 2*y_2)"],
    #         ["(2*yx4 ** 2)/y_1"],
    #         ["(2*yx2 ** 2*yx7)/(y_0*y_2 ** 2)"],
    #         ["(2*yx2*yx4*yx7)/(y_0*y_1*y_2)"],
    #         ["(2*yx4 ** 2*yx7)/(y_0*y_1 ** 2)"],
    #         ["(2*yx2*yx7 ** 2)/(y_0 ** 2*y_2)"],
    #         ["(2*yx4*yx7 ** 2)/(y_0 ** 2*y_1)"],
    #         ["(2*yx7 ** 2)/y_0"]
    #     ])
    #
    #     central_from_raw_exprs = to_sympy_matrix(
    #         [["x_0_0_2-y_2**2"],
    #         ["x_0_1_1-y_1*y_2"],
    #         ["x_0_2_0-y_1**2"],
    #         ["x_1_0_1-y_0*y_2"],
    #         ["x_1_1_0-y_0*y_1"],
    #         ["x_2_0_0-y_0**2"],
    #         ["-3*x_0_0_2*y_2+x_0_0_3+2*y_2**3"],
    #         ["-x_0_0_2*y_1-2*x_0_1_1*y_2+x_0_1_2+2*y_1*y_2**2"],
    #         ["-2*x_0_1_1*y_1-x_0_2_0*y_2+x_0_2_1+2*y_1**2*y_2"],
    #         ["-3*x_0_2_0*y_1+x_0_3_0+2*y_1**3"],
    #         ["-x_0_0_2*y_0-2*x_1_0_1*y_2+x_1_0_2+2*y_0*y_2**2"],
    #         ["-x_0_1_1*y_0-x_1_0_1*y_1-x_1_1_0*y_2+x_1_1_1+2*y_0*y_1*y_2"],
    #         ["-x_0_2_0*y_0-2*x_1_1_0*y_1+x_1_2_0+2*y_0*y_1**2"],
    #         ["-2*x_1_0_1*y_0-x_2_0_0*y_2+x_2_0_1+2*y_0**2*y_2"],
    #         ["-2*x_1_1_0*y_0-x_2_0_0*y_1+x_2_1_0+2*y_0**2*y_1"],
    #         ["-3*x_2_0_0*y_0+x_3_0_0+2*y_0**3"]
    #     ])
    #
    #     closed_raw_moms = to_sympy_matrix([
    #         ["y_2 ** 2+yx2"],
    #         ["y_1*y_2+(yx2*yx4)/(y_1*y_2)"],
    #         ["y_1 ** 2+yx4"],
    #         ["y_0*y_2+(yx2*yx7)/(y_0*y_2)"],
    #         ["y_0*y_1+(yx4*yx7)/(y_0*y_1)"],
    #         ["y_0 ** 2+yx7"],
    #         ["(2*yx2 ** 2)/y_2+3*y_2*yx2+y_2 ** 3"],
    #         ["y_1*yx2+y_1*y_2 ** 2+(2*yx2*yx4)/y_1+(2*yx2 ** 2*yx4)/(y_1*y_2 ** 2)"],
    #         ["y_2*yx4+y_1 ** 2*y_2+(2*yx2*yx4)/y_2+(2*yx2*yx4 ** 2)/(y_1 ** 2*y_2)"],
    #         ["(2*yx4 ** 2)/y_1+3*y_1*yx4+y_1 ** 3"],
    #         ["y_0*yx2+y_0*y_2 ** 2+(2*yx2*yx7)/y_0+(2*yx2 ** 2*yx7)/(y_0*y_2 ** 2)"],
    #         ["y_0*y_1*y_2+(y_0*yx2*yx4)/(y_1*y_2)+(y_1*yx2*yx7)/(y_0*y_2)+(y_2*yx4*yx7)/(y_0*y_1)+(2*yx2*yx4*yx7)/(y_0*y_1*y_2)"],
    #         ["y_0*yx4+y_0*y_1 ** 2+(2*yx4*yx7)/y_0+(2*yx4 ** 2*yx7)/(y_0*y_1 ** 2)"],
    #         ["y_2*yx7+y_0 ** 2*y_2+(2*yx2*yx7)/y_2+(2*yx2*yx7 ** 2)/(y_0 ** 2*y_2)"],
    #         ["y_1*yx7+y_0 ** 2*y_1+(2*yx4*yx7)/y_1+(2*yx4*yx7 ** 2)/(y_0 ** 2*y_1)"],
    #         ["(2*yx7 ** 2)/y_0+3*y_0*yx7+y_0 ** 3"]
    #     ])
    #     k_counter = [
    #         Moment([0, 0, 0], symbol=sympy.Integer(1)),
    #         Moment([1, 0, 0], symbol=sympy.Symbol("y_0")),
    #         Moment([0, 1, 0], symbol=sympy.Symbol("y_1")),
    #         Moment([0, 0, 1], symbol=sympy.Symbol("y_2")),
    #         Moment([0, 0, 2], symbol=sympy.Symbol("x_0_0_2")),
    #         Moment([0, 1, 1], symbol=sympy.Symbol("x_0_1_1")),
    #         Moment([0, 2, 0], symbol=sympy.Symbol("x_0_2_0")),
    #         Moment([1, 0, 1], symbol=sympy.Symbol("x_1_0_1")),
    #         Moment([1, 1, 0], symbol=sympy.Symbol("x_1_1_0")),
    #         Moment([2, 0, 0], symbol=sympy.Symbol("x_2_0_0")),
    #         Moment([0, 0, 3], symbol=sympy.Symbol("x_0_0_3")),
    #         Moment([0, 1, 2], symbol=sympy.Symbol("x_0_1_2")),
    #         Moment([0, 2, 1], symbol=sympy.Symbol("x_0_2_1")),
    #         Moment([0, 3, 0], symbol=sympy.Symbol("x_0_3_0")),
    #         Moment([1, 0, 2], symbol=sympy.Symbol("x_1_0_2")),
    #         Moment([1, 1, 1], symbol=sympy.Symbol("x_1_1_1")),
    #         Moment([1, 2, 0], symbol=sympy.Symbol("x_1_2_0")),
    #         Moment([2, 0, 1], symbol=sympy.Symbol("x_2_0_1")),
    #         Moment([2, 1, 0], symbol=sympy.Symbol("x_2_1_0")),
    #         Moment([3, 0, 0], symbol=sympy.Symbol("x_3_0_0"))
    #     ]
    #
    #     n_moments = 3
    #     closer = GammaCloser(n_moments, type=1)
    #     answer = closer.compute_closed_central_moments(central_from_raw_exprs,self.__n_counter, k_counter)
    #     self.assertTrue(sympy_expressions_equal(answer, expected))
