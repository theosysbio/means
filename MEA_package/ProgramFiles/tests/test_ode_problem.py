import unittest
import numpy as np
from numpy.testing import assert_array_equal
import sympy
from ode_problem import ODEProblem

class TestODEProblem(unittest.TestCase):

    def test_ode_rhs_as_function(self):
        """
        Given an ODEProblem with well specified LHS, RHS expressions as well as list of constants,
        the value of rhs_as_function given the appropriate params should be the same as the value of
        rhs evaluated for these params.
        :return:
        """
        lhs = sympy.Matrix(['y_1', 'y_2', 'y_3'])
        rhs = sympy.Matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1'])
        p = ODEProblem(lhs, rhs, constants=sympy.symbols(['c_1', 'c_2', 'c_3']), moments=["1,0,0", "0,1,0", "0,0,1"])

        rhs_as_function = p.right_hand_side_as_function([1, 2, 3])

        params = [4, 5, 6]  # y_1, y_2, y_3 in that order
        expected_ans = np.array([[11], [14], [7]])
        actual_ans = np.array(rhs_as_function(params))
        assert_array_equal(actual_ans, expected_ans)