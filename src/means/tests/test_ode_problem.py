import unittest

import numpy as np
from numpy.testing import assert_array_equal
import sympy

from means.approximation.ode_problem import ODEProblem, Moment, VarianceTerm


class TestODEProblem(unittest.TestCase):

    def test_ode_rhs_as_function(self):
        """
        Given an ODEProblem with well specified LHS, RHS expressions as well as list of constants,
        the value of rhs_as_function given the appropriate params should be the same as the value of
        rhs evaluated for these params.
        :return:
        """
        lhs = [Moment(np.ones(3),i) for i in sympy.Matrix(['y_1', 'y_2', 'y_3'])]
        rhs = sympy.Matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1'])

        p = ODEProblem('MEA', lhs, rhs, constants=sympy.symbols(['c_1', 'c_2', 'c_3']))

        rhs_as_function = p.right_hand_side_as_function([1, 2, 3])

        params = [4, 5, 6]  # y_1, y_2, y_3 in that order
        expected_ans = np.array([[11], [14], [7]])
        actual_ans = np.array(rhs_as_function(params))
        assert_array_equal(actual_ans, expected_ans)

    def test_ode_rhs_as_function_cache_does_not_persist_between_instances(self):
        """
        Given two ODEProblems, the cache should not persist between these objects.
        :return:
        """
        p1_lhs = [Moment(np.ones(3), i) for i in sympy.Matrix(['y_1', 'y_2', 'y_3'])]
        p1_rhs = sympy.Matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1'])

        p2_lhs = [Moment(np.ones(3), i) for i in sympy.Matrix(['y_1', 'y_2', 'y_3'])]
        p2_rhs = sympy.Matrix(['y_1', 'c_1', 'y_2+y_3'])

        p1 = ODEProblem('MEA', p1_lhs, p1_rhs, constants=sympy.symbols(['c_1', 'c_2', 'c_3']))
        p1_rhs_as_function = p1.right_hand_side_as_function([1, 2, 3])

        params = [4, 5, 6]  # y_1, y_2, y_3 in that order

        p2 = ODEProblem('MEA', p2_lhs, p2_rhs, constants=sympy.symbols(['c_1', 'c_2', 'c_3']))
        p2_rhs_as_function = p2.right_hand_side_as_function([1, 2, 3])

        p1_expected_ans = np.array([[11], [14], [7]])
        p2_expected_ans = np.array([[4], [1], [6+5]])
        p1_actual_ans = np.array(p1_rhs_as_function(params))
        p2_actual_ans = np.array(p2_rhs_as_function(params))

        assert_array_equal(p1_actual_ans, p1_expected_ans)
        assert_array_equal(p2_actual_ans, p2_expected_ans)



    def test_ode_moment_no_description_from_variance_terms(self):
        """
        Given  Variance terms as left hand side terms, the generated descriptions
        dict should have nones
        for each of the symbols
        """
        lhs = [VarianceTerm(i) for i in ['V34', 'V32', 'V11']]
        rhs = sympy.Matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1'])
        p = ODEProblem('LNA', lhs, rhs, constants=sympy.symbols(['c_1', 'c_2', 'c_3']))

        for i,l in enumerate(lhs):
            self.assertIsNone(p.descriptions_dict[l.symbol].descriptor)


    def test_ode_moment_getting_n_vector_from_dict_and_key(self):
        """
        Given a list of descriptor and a list of symbols used to create Moment,
        Then problem description_dict should return a numpy array equal to the descriptor
        for each corresponding symbol
        :return:
        """
        symbs = sympy.Matrix(['y_1', 'y_2', 'y_3'])
        desc = [[0,0,1],[1,0,432],[21,43,34]]
        lhs = [Moment(d,s) for d,s in zip(desc,symbs)]
        rhs = sympy.Matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1'])
        p = ODEProblem('MEA', lhs, rhs, constants=sympy.symbols(['c_1', 'c_2', 'c_3']))
        for i,l in enumerate(lhs):
            self.assertEqual((p.descriptions_dict[l.symbol].descriptor == np.array(desc[i])).all(), True )

