import unittest
import sympy
from raw_to_central import raw_to_central
from ode_problem import Moment
class TestRawToCentral(unittest.TestCase):

    def test_a_two_species_problem(self):
        """
        Given a single n vector - a single mixed moment to compute values for,
        the function should return a single correct equation for that moment's value

        :return:
        """
        ymat = sympy.Matrix(["y_0","y_1","y_2"])
        counter_nvecs = [[0, 0], [0, 2], [1, 1], [2, 0]]
        mcounter_nvecs = [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]]

        counter = [Moment(c,sympy.Symbol("YU{0}".format(i))) for i,c in enumerate(counter_nvecs)]
        mcounter = [Moment(c,sympy.Symbol("y_{0}".format(i))) for i,c in enumerate(mcounter_nvecs)]


        answer = raw_to_central(counter, ymat, mcounter)

        expected =  sympy.Matrix([
            ["y_0*y_1**2 - 2*y_1**2 + y_3"],
            ["y_0**2*y_1 - y_0*y_1 - y_1*y_2 + y_4"],
            ["y_0**3 - 2*y_0*y_2 + y_5"]])
        self.assertEqual(answer, expected)

    def test_a_two_species_problem(self):
        """
        Given a single n vector - a single mixed moment to compute values for,
        the function should return a single correct equation for that moment's value

        :return:
        """
        ymat = sympy.Matrix(["y_0","y_1"])
        counter_nvecs = [[0, 0], [0, 2], [1, 1], [2, 0]]
        mcounter_nvecs = [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]]

        counter = [Moment(c,sympy.Symbol("YU{0}".format(i))) for i,c in enumerate(counter_nvecs)]
        mcounter = [Moment(c,sympy.Symbol("y_{0}".format(i))) for i,c in enumerate(mcounter_nvecs)]


        answer = raw_to_central(counter, ymat, mcounter)

        expected =  sympy.Matrix([
            ["y_0*y_1**2 - 2*y_1**2 + y_3"],
            ["y_0**2*y_1 - y_0*y_1 - y_1*y_2 + y_4"],
            ["y_0**3 - 2*y_0*y_2 + y_5"]])
        self.assertEqual(answer, expected)


    def test_a_three_species_third_order_problem(self):
       counter = [Moment([0, 0, 0] ,  sympy.Integer(1)), Moment([0, 0, 2] ,  sympy.Symbol("yx1")), Moment([0, 0, 3] ,  sympy.Symbol("yx2")), Moment([0, 1, 1] ,  sympy.Symbol("yx3")), Moment([0, 1, 2] ,  sympy.Symbol("yx4")), Moment([0, 2, 0] ,  sympy.Symbol("yx5")), Moment([0, 2, 1] ,  sympy.Symbol("yx6")), Moment([0, 3, 0] ,  sympy.Symbol("yx7")), Moment([1, 0, 1] ,  sympy.Symbol("yx8")), Moment([1, 0, 2] ,  sympy.Symbol("yx9")), Moment([1, 1, 0] ,  sympy.Symbol("yx10")), Moment([1, 1, 1] ,  sympy.Symbol("yx11")), Moment([1, 2, 0] ,  sympy.Symbol("yx12")), Moment([2, 0, 0] ,  sympy.Symbol("yx13")), Moment([2, 0, 1] ,  sympy.Symbol("yx14")), Moment([2, 1, 0] ,  sympy.Symbol("yx15")), Moment([3, 0, 0] ,  sympy.Symbol("yx16"))]
       ymat = sympy.Matrix(["y_0","y_1","y_2"])
       mcounter = [Moment([0, 0, 0] ,  sympy.Integer(1)), Moment([0, 0, 1] ,  sympy.Symbol("y_2")), Moment([0, 0, 2] ,  sympy.Symbol("x_0_0_2")), Moment([0, 0, 3] ,  sympy.Symbol("x_0_0_3")), Moment([0, 1, 0] ,  sympy.Symbol("y_1")), Moment([0, 1, 1] ,  sympy.Symbol("x_0_1_1")), Moment([0, 1, 2] ,  sympy.Symbol("x_0_1_2")), Moment([0, 2, 0] ,  sympy.Symbol("x_0_2_0")), Moment([0, 2, 1] ,  sympy.Symbol("x_0_2_1")), Moment([0, 3, 0] ,  sympy.Symbol("x_0_3_0")), Moment([1, 0, 0] ,  sympy.Symbol("y_0")), Moment([1, 0, 1] ,  sympy.Symbol("x_1_0_1")), Moment([1, 0, 2] ,  sympy.Symbol("x_1_0_2")), Moment([1, 1, 0] ,  sympy.Symbol("x_1_1_0")), Moment([1, 1, 1] ,  sympy.Symbol("x_1_1_1")), Moment([1, 2, 0] ,  sympy.Symbol("x_1_2_0")), Moment([2, 0, 0] ,  sympy.Symbol("x_2_0_0")), Moment([2, 0, 1] ,  sympy.Symbol("x_2_0_1")), Moment([2, 1, 0] ,  sympy.Symbol("x_2_1_0")), Moment([3, 0, 0] ,  sympy.Symbol("x_3_0_0"))]

       answer = sympy.Matrix(raw_to_central(counter, ymat, mcounter))
       expected = sympy.Matrix(
        [["                                                     1*y_2**2 + x_0_0_2 - 2*y_2**2"],
        ["                                    -1*y_2**3 - 3*x_0_0_2*y_2 + x_0_0_3 + 3*y_2**3"],
        ["                                                   1*y_1*y_2 + x_0_1_1 - 2*y_1*y_2"],
        ["              -1*y_1*y_2**2 - x_0_0_2*y_1 - 2*x_0_1_1*y_2 + x_0_1_2 + 3*y_1*y_2**2"],
        ["                                                     1*y_1**2 + x_0_2_0 - 2*y_1**2"],
        ["              -1*y_1**2*y_2 - 2*x_0_1_1*y_1 - x_0_2_0*y_2 + x_0_2_1 + 3*y_1**2*y_2"],
        ["                                    -1*y_1**3 - 3*x_0_2_0*y_1 + x_0_3_0 + 3*y_1**3"],
        ["                                                   1*y_0*y_2 + x_1_0_1 - 2*y_0*y_2"],
        ["              -1*y_0*y_2**2 - x_0_0_2*y_0 - 2*x_1_0_1*y_2 + x_1_0_2 + 3*y_0*y_2**2"],
        ["                                                   1*y_0*y_1 + x_1_1_0 - 2*y_0*y_1"],
        ["-1*y_0*y_1*y_2 - x_0_1_1*y_0 - x_1_0_1*y_1 - x_1_1_0*y_2 + x_1_1_1 + 3*y_0*y_1*y_2"],
        ["              -1*y_0*y_1**2 - x_0_2_0*y_0 - 2*x_1_1_0*y_1 + x_1_2_0 + 3*y_0*y_1**2"],
        ["                                                     1*y_0**2 + x_2_0_0 - 2*y_0**2"],
        ["              -1*y_0**2*y_2 - 2*x_1_0_1*y_0 - x_2_0_0*y_2 + x_2_0_1 + 3*y_0**2*y_2"],
        ["              -1*y_0**2*y_1 - 2*x_1_1_0*y_0 - x_2_0_0*y_1 + x_2_1_0 + 3*y_0**2*y_1"],
        ["                                    -1*y_0**3 - 3*x_2_0_0*y_0 + x_3_0_0 + 3*y_0**3"]])

       self.assertEqual(answer, expected)

