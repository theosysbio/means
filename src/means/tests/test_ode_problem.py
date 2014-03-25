import unittest

import numpy as np
from numpy.testing import assert_array_equal
import sympy

from means.core import ODEProblem, Moment, Moments
from means.util.sympyhelpers import to_sympy_matrix


class TestODEProblem(unittest.TestCase):

    def test_ode_rhs_as_function(self):
        """
        Given an ODEProblem with well specified LHS, RHS expressions as well as list of constants,
        the value of rhs_as_function given the appropriate params should be the same as the value of
        rhs evaluated for these params. The returned answer should also be an one-dimensional numpy array.
        :return:
        """
        lhs = [Moment(np.ones(3),i) for i in sympy.Matrix(['y_1', 'y_2', 'y_3'])]
        rhs = to_sympy_matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1'])

        p = ODEProblem('MEA', lhs, rhs, parameters=sympy.symbols(['c_1', 'c_2', 'c_3']))

        rhs_as_function = p.right_hand_side_as_function

        values = [4, 5, 6]  # y_1, y_2, y_3 in that order
        expected_ans = np.array([11, 14, 7])
        actual_ans = np.array(rhs_as_function(values, [1, 2, 3]))
        self.assertEqual(actual_ans.ndim, 1)  # Returned answer must be an one-dimensional array,
                                              # otherwise ExplicitEuler solver would fail.
        assert_array_equal(actual_ans, expected_ans)

    def _check_ode_rhs_as_function_ans(self, p1_rhs_as_function, p2_rhs_as_function):

        constants = [1, 2, 3]
        values_p1 = [4, 5, 6, 5]  # y_1, y_2, y_3, y_4 in that order
        values_p2 = [4, 5, 6]  # y_1, y_2, y_3  in that order

        p1_expected_ans = np.array([11, 14, 7, 8])
        p2_expected_ans = np.array([4, 1, 6+5])
        p1_actual_ans = np.array(p1_rhs_as_function(values_p1, constants))
        p2_actual_ans = np.array(p2_rhs_as_function(values_p2, constants))

        # This checks if by any means p2 is able to "override" the p1 result
        p1_ans_after_p2 = np.array(p1_rhs_as_function(values_p1, constants))

        assert_array_equal(p1_actual_ans, p1_expected_ans)
        assert_array_equal(p2_actual_ans, p2_expected_ans)
        assert_array_equal(p1_ans_after_p2, p1_expected_ans)


    def test_ode_rhs_as_function_cache_does_not_persist_between_instances(self):
        """
        Given two ODEProblems, the cache should not persist between these objects.
        :return:
        """

        p1_lhs = [Moment(np.ones(4), i) for i in sympy.Matrix(['y_1', 'y_2', 'y_3', 'y_4'])]
        p1_rhs = to_sympy_matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1', 'y_1*2'])

        p2_lhs = [Moment(np.ones(3), i) for i in sympy.Matrix(['y_1', 'y_2', 'y_3'])]
        p2_rhs = to_sympy_matrix(['y_1', 'c_1', 'y_2+y_3'])

        p1 = ODEProblem('MEA', p1_lhs, p1_rhs, parameters=sympy.symbols(['c_1', 'c_2', 'c_3']))
        p1_rhs_as_function = p1.right_hand_side_as_function

        p2 = ODEProblem('MEA', p2_lhs, p2_rhs, parameters=sympy.symbols(['c_1', 'c_2', 'c_3']))
        p2_rhs_as_function = p2.right_hand_side_as_function

        constants = [1, 2, 3]
        values_p1 = [4, 5, 6, 5]  # y_1, y_2, y_3, y_4 in that order
        values_p2 = [4, 5, 6]  # y_1, y_2, y_3  in that order

        p1_expected_ans = np.array([11, 14, 7, 8])
        p2_expected_ans = np.array([4, 1, 6+5])
        p1_actual_ans = np.array(p1_rhs_as_function(values_p1, constants))
        p2_actual_ans = np.array(p2_rhs_as_function(values_p2, constants))

        # This checks if by any means p2 is able to "override" the p1 result
        p1_ans_after_p2 = np.array(p1_rhs_as_function(values_p1, constants))

        assert_array_equal(p1_actual_ans, p1_expected_ans)
        assert_array_equal(p2_actual_ans, p2_expected_ans)
        assert_array_equal(p1_ans_after_p2, p1_expected_ans)


    def test_ode_moment_no_description_from_variance_terms(self):
        """
        Given  Variance terms as left hand side terms, the generated descriptions
        dict should have nones
        for each of the symbols
        """
        lhs = [Moments(pos, term) for term, pos in [('V34', (3, 4)), ('V32', (3, 2)), ('V11', (1, 1))]]
        rhs = to_sympy_matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1'])
        p = ODEProblem('LNA', lhs, rhs, parameters=sympy.symbols(['c_1', 'c_2', 'c_3']))

        for i,l in enumerate(lhs):
            self.assertIsNone(p._descriptions_dict[l.symbol].descriptor)


    def test_ode_moment_getting_n_vector_from_dict_and_key(self):
        """
        Given a list of descriptor and a list of symbols used to create Moment,
        Then problem descriptor_for_symbol function should return the correct
        descriptor for each corresponding symbol
        :return:
        """
        symbs = to_sympy_matrix(['y_1', 'y_2', 'y_3'])
        desc = [[0, 0, 1], [1, 0, 432], [21, 43, 34]]
        lhs = [Moment(d, s) for d, s in zip(desc, symbs)]
        rhs = to_sympy_matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1'])
        p = ODEProblem('MEA', lhs, rhs, parameters=sympy.symbols(['c_1', 'c_2', 'c_3']))
        for i, l in enumerate(lhs):
            self.assertEqual(p.descriptor_for_symbol(l.symbol), l)

def _get_rhs_as_function(args):
    lhs, rhs = args

    p = ODEProblem('MEA', lhs, rhs, parameters=sympy.symbols(['c_1', 'c_2', 'c_3']))
    # access right_hand_side_as_function
    rhs_as_function = p.right_hand_side_as_function

    return p