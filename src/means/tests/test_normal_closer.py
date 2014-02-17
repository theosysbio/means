import unittest
import sympy
from means.approximation.mea.normal_closer import NormalCloser
from means.approximation.ode_problem import Moment
from means.util.sympyhelpers import sympy_expressions_equal
from means.util.sympyhelpers import to_sympy_matrix

class TestNormalCloser(unittest.TestCase):

    __problem_moments = [
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
    # def test_partition(self):
    #
    #     test_list_for_partition = [sympy.Symbol('A'),sympy.Symbol('B'),sympy.Symbol('C'),sympy.Symbol('D')]
    #     expected = [[[sympy.Symbol('A'),sympy.Symbol('B')],[sympy.Symbol('A'),sympy.Symbol('C')]],
    #                 [[sympy.Symbol('A'),sympy.Symbol('D')],[sympy.Symbol('B'),sympy.Symbol('C')]],
    #                 [[sympy.Symbol('B'),sympy.Symbol('D')],[sympy.Symbol('C'),sympy.Symbol('D')]]]
    #
    #
    #     closer = NormalCloser(3,multivariate=True)
    #     answer = closer.partition(2,[[]],0,test_list_for_partition)
    #
    #     self.assertEqual(answer, expected)


    def test_log_normal_closer_wrapper(self):

        mfk = to_sympy_matrix([
                    ["-(c_2*c_6*y_2*yx17*(c_6+y_0)**6+c_2*c_6*yx5*(c_6+y_0)**8+c_2*y_0*y_2*(c_6+y_0)**9-(c_0-c_1*y_0)*(c_6+y_0)**10-(c_6+y_0)**7*(c_2*c_6*y_2*yx7+c_2*c_6*yx15))/(c_6+y_0)**10"],
                    ["c_3*y_0-c_4*y_1"],
                    ["c_4*y_1-c_5*y_2"],
                    ["c_4*y_1+2*c_4*yx3+c_5*y_2-2*c_5*yx2"],
                    ["c_3*yx5-c_4*y_1+c_4*yx4-yx3*(c_4+c_5)"],
                    ["c_3*y_0+2*c_3*yx6+c_4*y_1-2*c_4*yx4"],
                    ["c_2*c_6*y_2*yx15/(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)-c_2*y_0*yx2/(c_6+y_0)+c_2*yx12*(y_0/(c_6+y_0)-1)/(c_6+y_0)+c_4*yx6-yx5*(-c_2*c_6*y_2-2*c_2*y_0*y_2+2*c_2*y_2*(c_6+y_0)+(c_1+c_5)*(c_6+y_0)**2)/(c_6+y_0)**2"],
                    ["-(c_2*c_6*yx13*(c_6+y_0)**4+c_2*y_0*yx3*(c_6+y_0)**5-c_3*yx7*(c_6+y_0)**6-(c_6+y_0)**3*(c_2*c_6*y_2*yx16-yx6*(-c_2*y_0*y_2*(c_6+y_0)+c_2*y_2*(c_6+y_0)**2+(c_1+c_4)*(c_6+y_0)**3)))/(c_6+y_0)**6"],
                    ["c_2*c_6*y_2*yx17*(2*c_6+2*y_0+1)/(c_6**4+4*c_6**3*y_0+6*c_6**2*y_0**2+4*c_6*y_0**3+y_0**4)-c_2*c_6*yx15*(-2*c_6-2*y_0-1)/(-c_6**3-3*c_6**2*y_0-3*c_6*y_0**2-y_0**3)+c_2*yx5*(-2*c_6*y_0+c_6-2*y_0**2)/(c_6**2+2*c_6*y_0+y_0**2)+yx7*(2*c_1*c_6**3+6*c_1*c_6**2*y_0+6*c_1*c_6*y_0**2+2*c_1*y_0**3+2*c_2*c_6**2*y_2+2*c_2*c_6*y_0*y_2+c_2*c_6*y_2)/(-c_6**3-3*c_6**2*y_0-3*c_6*y_0**2-y_0**3)+(c_0*c_6+c_0*y_0+c_1*c_6*y_0+c_1*y_0**2+c_2*y_0*y_2)/(c_6+y_0)"],
                    ["c_4*y_1+3*c_4*yx3+3*c_4*yx9-c_5*y_2+3*c_5*yx2-3*c_5*yx8"],
                    ["c_3*yx12-c_4*y_1+2*c_4*yx10-2*c_4*yx3+c_4*yx4-c_4*yx9+c_5*yx3-2*c_5*yx9"],
                    ["2*c_3*yx13+c_3*yx5+c_4*y_1-2*c_4*yx10+c_4*yx11+c_4*yx3-2*c_4*yx4-c_5*yx10"],
                    ["c_3*y_0+3*c_3*yx14+3*c_3*yx6-c_4*y_1-3*c_4*yx11+3*c_4*yx4"],
                    ["c_2*c_6*y_2*yx17*yx2/(c_6**4+4*c_6**3*y_0+6*c_6**2*y_0**2+4*c_6*y_0**3+y_0**4)-c_2*c_6*y_2*yx2*yx7/(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)-c_2*c_6*yx15*yx2/(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)-c_2*y_0*yx8/(c_6+y_0)+2*c_4*yx13+c_4*yx6-yx12*(-2*c_2*c_6*y_2-3*c_2*y_0*y_2+3*c_2*y_2*(c_6+y_0)+(c_1+2*c_5)*(c_6+y_0)**2)/(c_6+y_0)**2-yx2*(c_2*y_0*y_2-(c_0-c_1*y_0)*(c_6+y_0))/(c_6+y_0)+yx5*(c_2*c_6*yx2+2*c_4*c_6**2*y_1+4*c_4*c_6*y_0*y_1+2*c_4*y_0**2*y_1-2*c_5*c_6**2*y_2+c_5*c_6**2-4*c_5*c_6*y_0*y_2+2*c_5*c_6*y_0-2*c_5*y_0**2*y_2+c_5*y_0**2)/(c_6**2+2*c_6*y_0+y_0**2)+(-c_0*c_6*yx2-c_0*y_0*yx2+c_1*c_6*y_0*yx2+c_1*y_0**2*yx2+c_2*y_0*y_2*yx2-2*c_4*c_6*y_1*yx5-2*c_4*y_0*y_1*yx5+2*c_5*c_6*y_2*yx5+2*c_5*y_0*y_2*yx5)/(c_6+y_0)"],
                    ["c_2*c_6*y_2*yx17*yx3/(c_6**4+4*c_6**3*y_0+6*c_6**2*y_0**2+4*c_6*y_0**3+y_0**4)-c_2*c_6*y_2*yx3*yx7/(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)-c_2*y_0*yx9/(c_6+y_0)+c_4*yx14-yx13*(-c_2*c_6*y_2-2*c_2*y_0*y_2+2*c_2*y_2*(c_6+y_0)+(c_6+y_0)**2*(c_1+c_4+c_5))/(c_6+y_0)**2+yx15*(-c_2*c_6*yx3+c_3*c_6**3+3*c_3*c_6**2*y_0+3*c_3*c_6*y_0**2+c_3*y_0**3)/(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)-yx3*(-c_0*c_6-c_0*y_0+c_1*c_6*y_0+c_1*y_0**2+c_2*y_0*y_2)/(c_6+y_0)+yx5*(c_2*c_6*yx3+c_3*c_6**2*y_0+2*c_3*c_6*y_0**2+c_3*y_0**3-c_4*c_6**2*y_1-2*c_4*c_6*y_0*y_1-c_4*y_0**2*y_1)/(c_6**2+2*c_6*y_0+y_0**2)-yx6*(-c_4*y_1+c_4+c_5*y_2)+(-c_0*c_6*yx3-c_0*y_0*yx3+c_1*c_6*y_0*yx3+c_1*y_0**2*yx3+c_2*y_0*y_2*yx3-c_3*c_6*y_0*yx5-c_3*y_0**2*yx5+c_4*c_6*y_1*yx5-c_4*c_6*y_1*yx6+c_4*y_0*y_1*yx5-c_4*y_0*y_1*yx6+c_5*c_6*y_2*yx6+c_5*y_0*y_2*yx6)/(c_6+y_0)"],
                    ["c_2*c_6*y_2*yx17*yx4/(c_6**4+4*c_6**3*y_0+6*c_6**2*y_0**2+4*c_6*y_0**3+y_0**4)-c_2*c_6*yx15*yx4/(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)+c_2*c_6*yx4*yx5/(c_6**2+2*c_6*y_0+y_0**2)-c_2*y_0*yx10/(c_6+y_0)+2*c_3*yx16-yx14*(-c_2*y_0*y_2+c_2*y_2*(c_6+y_0)+(c_1+2*c_4)*(c_6+y_0)**2)/(c_6+y_0)**2-yx4*(c_2*y_0*y_2-(c_0-c_1*y_0)*(c_6+y_0))/(c_6+y_0)+yx6*(2*c_3*y_0-2*c_4*y_1+c_4)+yx7*(-c_2*c_6*y_2*yx4+c_3*c_6**3+3*c_3*c_6**2*y_0+3*c_3*c_6*y_0**2+c_3*y_0**3)/(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)+(-c_0*c_6*yx4-c_0*y_0*yx4+c_1*c_6*y_0*yx4+c_1*y_0**2*yx4+c_2*y_0*y_2*yx4-2*c_3*c_6*y_0*yx6-2*c_3*y_0**2*yx6+2*c_4*c_6*y_1*yx6+2*c_4*y_0*y_1*yx6)/(c_6+y_0)"],
                    ["2*c_2*c_6*y_2*yx17*yx5/(c_6**4+4*c_6**3*y_0+6*c_6**2*y_0**2+4*c_6*y_0**3+y_0**4)+c_2*y_0*yx2/(c_6+y_0)-c_2*yx12*(2*c_6*y_0-c_6+2*y_0**2)/(c_6**2+2*c_6*y_0+y_0**2)+c_4*yx16-yx15*(2*c_1*c_6**3+6*c_1*c_6**2*y_0+6*c_1*c_6*y_0**2+2*c_1*y_0**3+2*c_2*c_6**2*y_2+2*c_2*c_6*y_0*y_2+c_2*c_6*y_2+2*c_2*c_6*yx5+c_5*c_6**3+3*c_5*c_6**2*y_0+3*c_5*c_6*y_0**2+c_5*y_0**3)/(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)+yx5*(2*c_0*c_6**2+4*c_0*c_6*y_0+2*c_0*y_0**2-2*c_1*c_6**2*y_0+c_1*c_6**2-4*c_1*c_6*y_0**2+2*c_1*c_6*y_0-2*c_1*y_0**3+c_1*y_0**2-2*c_2*c_6*y_0*y_2+c_2*c_6*y_2+2*c_2*c_6*yx5-2*c_2*y_0**2*y_2)/(c_6**2+2*c_6*y_0+y_0**2)-yx7*(2*c_2*c_6*y_2*yx5-c_4*c_6**3*y_1-3*c_4*c_6**2*y_0*y_1-3*c_4*c_6*y_0**2*y_1-c_4*y_0**3*y_1+c_5*c_6**3*y_2+3*c_5*c_6**2*y_0*y_2+3*c_5*c_6*y_0**2*y_2+c_5*y_0**3*y_2)/(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)+(-2*c_0*c_6*yx5-2*c_0*y_0*yx5+2*c_1*c_6*y_0*yx5+2*c_1*y_0**2*yx5+2*c_2*y_0*y_2*yx5-c_4*c_6*y_1*yx7-c_4*y_0*y_1*yx7+c_5*c_6*y_2*yx7+c_5*y_0*y_2*yx7)/(c_6+y_0)"],
                    ["-2*c_2*c_6*yx15*yx6/(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)+2*c_2*c_6*yx5*yx6/(c_6**2+2*c_6*y_0+y_0**2)+c_2*y_0*yx3/(c_6+y_0)-c_2*yx13*(2*c_6*y_0-c_6+2*y_0**2)/(c_6**2+2*c_6*y_0+y_0**2)-yx16*(2*c_2*c_6*y_0*y_2+2*c_2*y_0**2*y_2-c_2*y_0*y_2+2*c_2*y_2*(c_6+y_0)**2+(2*c_1+c_4)*(c_6+y_0)**3-(c_6+y_0)*(4*c_2*y_0*y_2-c_2*y_2))/(c_6+y_0)**3+yx17*(2*c_2*c_6*y_2*yx6+c_3*c_6**4+4*c_3*c_6**3*y_0+6*c_3*c_6**2*y_0**2+4*c_3*c_6*y_0**3+c_3*y_0**4)/(c_6**4+4*c_6**3*y_0+6*c_6**2*y_0**2+4*c_6*y_0**3+y_0**4)+yx6*(2*c_0-4*c_1*y_0+c_1+2*c_2*y_0**2*y_2/(c_6+y_0)**2-4*c_2*y_0*y_2/(c_6+y_0)-c_2*y_0*y_2/(c_6+y_0)**2+c_2*y_2/(c_6+y_0)-2*c_4*y_0+2*y_0*(c_1-c_2*y_0*y_2/(c_6+y_0)**2+c_2*y_2/(c_6+y_0)+c_4))-yx7*(2*c_2*c_6*y_2*yx6-c_3*c_6**3*y_0-3*c_3*c_6**2*y_0**2-3*c_3*c_6*y_0**3-c_3*y_0**4+c_4*c_6**3*y_1+3*c_4*c_6**2*y_0*y_1+3*c_4*c_6*y_0**2*y_1+c_4*y_0**3*y_1)/(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)+(-2*c_0*c_6*yx6-2*c_0*y_0*yx6+2*c_1*c_6*y_0*yx6+2*c_1*y_0**2*yx6+2*c_2*y_0*y_2*yx6-c_3*c_6*y_0*yx7-c_3*y_0**2*yx7+c_4*c_6*y_1*yx7+c_4*y_0*y_1*yx7)/(c_6+y_0)"],
                    ["-(-c_2*yx5*(c_6+y_0)*(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)*(3*c_6*y_0+3*c_6*yx7-c_6+3*y_0**2)*(c_6**4+4*c_6**3*y_0+6*c_6**2*y_0**2+4*c_6*y_0**3+y_0**4)+yx17*(c_6+y_0)*(c_6**2+2*c_6*y_0+y_0**2)*(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)*(3*c_1*c_6**4+12*c_1*c_6**3*y_0+18*c_1*c_6**2*y_0**2+12*c_1*c_6*y_0**3+3*c_1*y_0**4+3*c_2*c_6**3*y_2+6*c_2*c_6**2*y_0*y_2+3*c_2*c_6**2*y_2+3*c_2*c_6*y_0**2*y_2+3*c_2*c_6*y_0*y_2-3*c_2*c_6*y_2*yx7+c_2*c_6*y_2)-(c_6+y_0)*(-c_2*yx15*(3*c_6**2*y_0-3*c_6**2+6*c_6*y_0**2-3*c_6*y_0+3*c_6*yx7-c_6+3*y_0**3)+yx7*(3*c_0*c_6**3+9*c_0*c_6**2*y_0+9*c_0*c_6*y_0**2+3*c_0*y_0**3-3*c_1*c_6**3*y_0+3*c_1*c_6**3-9*c_1*c_6**2*y_0**2+9*c_1*c_6**2*y_0-9*c_1*c_6*y_0**3+9*c_1*c_6*y_0**2-3*c_1*y_0**4+3*c_1*y_0**3-3*c_2*c_6**2*y_0*y_2+3*c_2*c_6**2*y_2-6*c_2*c_6*y_0**2*y_2+3*c_2*c_6*y_0*y_2-3*c_2*c_6*y_2*yx7+c_2*c_6*y_2-3*c_2*y_0**3*y_2))*(c_6**2+2*c_6*y_0+y_0**2)*(c_6**4+4*c_6**3*y_0+6*c_6**2*y_0**2+4*c_6*y_0**3+y_0**4)-(c_6**2+2*c_6*y_0+y_0**2)*(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)*(c_6**4+4*c_6**3*y_0+6*c_6**2*y_0**2+4*c_6*y_0**3+y_0**4)*(-3*c_0*c_6*yx7+c_0*c_6-3*c_0*y_0*yx7+c_0*y_0+3*c_1*c_6*y_0*yx7-c_1*c_6*y_0+3*c_1*y_0**2*yx7-c_1*y_0**2+3*c_2*y_0*y_2*yx7-c_2*y_0*y_2))/((c_6+y_0)*(c_6**2+2*c_6*y_0+y_0**2)*(c_6**3+3*c_6**2*y_0+3*c_6*y_0**2+y_0**3)*(c_6**4+4*c_6**3*y_0+6*c_6**2*y_0**2+4*c_6*y_0**3+y_0**4))"]
            ])
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


        k_counter = [
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

        n_moments = 3
        species = to_sympy_matrix([["y_0"],["y_1"],["y_2"]])
        prob_moments = self.__problem_moments
        expected = sympy.Matrix([
            ["c_0-c_1*y_0-(c_2*c_6*yx5)/(c_6+y_0) ** 2-(c_2*y_0*y_2)/(c_6+y_0)+(c_2*c_6*y_2*yx7)/(c_6+y_0) ** 3"],
            ["c_3*y_0-c_4*y_1"],
            ["c_4*y_1-c_5*y_2"],
            ["c_4*y_1+c_5*y_2+2*c_4*yx3-2*c_5*yx2"],
            ["c_3*yx5-c_4*yx3-c_4*y_1+c_4*yx4-c_5*yx3"],
            ["c_3*y_0+c_4*y_1-2*c_4*yx4+2*c_3*yx6"],
            ["c_4*yx6-c_1*yx5-c_5*yx5-(c_2*y_0*yx2)/(c_6+y_0)-(c_2*y_2*yx5)/(c_6+y_0)+(c_2*y_0*y_2*yx5)/(c_6+y_0) ** 2"],
            ["c_3*yx7-c_1*yx6-c_4*yx6-(c_2*y_0*yx3)/(c_6+y_0)-(c_2*y_2*yx6)/(c_6+y_0)+(c_2*y_0*y_2*yx6)/(c_6+y_0) ** 2"],
            ["(c_0*c_6 ** 3+c_0*y_0 ** 3+c_1*y_0 ** 4+c_2*y_0 ** 3*y_2-2*c_2*y_0 ** 3*yx5-2*c_1*y_0 ** 3*yx7+3*c_1*c_6 ** 2*y_0 ** 2+3*c_0*c_6*y_0 ** 2+3*c_0*c_6 ** 2*y_0+3*c_1*c_6*y_0 ** 3+c_1*c_6 ** 3*y_0+c_2*c_6 ** 2*yx5-2*c_1*c_6 ** 3*yx7+c_2*c_6*y_0*yx5-c_2*c_6*y_2*yx7+2*c_2*c_6*y_0 ** 2*y_2+c_2*c_6 ** 2*y_0*y_2-4*c_2*c_6*y_0 ** 2*yx5-2*c_2*c_6 ** 2*y_0*yx5-6*c_1*c_6*y_0 ** 2*yx7-6*c_1*c_6 ** 2*y_0*yx7-2*c_2*c_6 ** 2*y_2*yx7-2*c_2*c_6*y_0*y_2*yx7)/(c_6+y_0) ** 3"]
        ])
        closer = NormalCloser(n_moments,multivariate=True)
        answer, lhs_answer = closer.parametric_closer_wrapper(mfk, central_from_raw_exprs, species, k_counter, prob_moments)

        self.assertTrue(sympy_expressions_equal(answer, expected))