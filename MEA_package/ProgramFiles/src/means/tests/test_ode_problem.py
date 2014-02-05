import unittest

import numpy as np
from numpy.testing import assert_array_equal
import sympy

from means.simulation.ode_problem import ODEProblem


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
        p = ODEProblem('MEA', lhs, rhs, constants=sympy.symbols(['c_1', 'c_2', 'c_3']), description_of_lhs_terms=None)

        rhs_as_function = p.right_hand_side_as_function([1, 2, 3])

        params = [4, 5, 6]  # y_1, y_2, y_3 in that order
        expected_ans = np.array([[11], [14], [7]])
        actual_ans = np.array(rhs_as_function(params))
        assert_array_equal(actual_ans, expected_ans)


    def test_ode_moment_description_generation_none_should_generate_no_descriptions(self):
        """
        Given None for description of left hand side terms, the generated descriptions dict should have nones
        for each of the symbols
        """
        lhs = sympy.Matrix(['y_1', 'y_2', 'y_3'])
        rhs = sympy.Matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1'])
        p = ODEProblem('MEA', lhs, rhs, constants=sympy.symbols(['c_1', 'c_2', 'c_3']), description_of_lhs_terms=None)

        for i, symbol in enumerate(lhs):
            self.assertIsNone(p.descriptions_dict[symbol])
            self.assertIsNone(p.ordered_descriptions[i])

    def test_ode_moment_description_generation_from_string_keys(self):
        """
        Given string keys in the description dict, the generated moment descriptions should have appropriate
        descriptions generated.
        """

        lhs = sympy.Matrix(['y_1', 'y_2', 'y_3'])
        rhs = sympy.Matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1'])
        descriptions = {'y_1': 'foo', 'y_3': 'bar'}
        p = ODEProblem('MEA', lhs, rhs, constants=sympy.symbols(['c_1', 'c_2', 'c_3']), description_of_lhs_terms=descriptions)

        self.assertEquals(p.descriptions_dict[sympy.Symbol('y_1')], 'foo')
        self.assertIsNone(p.descriptions_dict[sympy.Symbol('y_2')])
        self.assertEquals(p.descriptions_dict[sympy.Symbol('y_3')], 'bar')

        self.assertEquals(p.ordered_descriptions[0], 'foo')
        self.assertIsNone(p.ordered_descriptions[1])
        self.assertEquals(p.ordered_descriptions[2], 'bar')


    def test_ode_moment_description_generation_from_symbol_keys(self):
        """
        Given symbol keys in the description dict, the generated moment descriptions should have appropriate
        descriptions generated.
        """

        lhs = sympy.Matrix(['y_1', 'y_2', 'y_3'])
        rhs = sympy.Matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1'])
        y_1 = sympy.Symbol('y_1')
        y_2 = sympy.Symbol('y_2')
        y_3 = sympy.Symbol('y_3')

        descriptions = {y_1: 'foo', y_3: 'bar'}
        p = ODEProblem('MEA',lhs, rhs, constants=sympy.symbols(['c_1', 'c_2', 'c_3']), description_of_lhs_terms=descriptions)

        self.assertEquals(p.descriptions_dict[y_1], 'foo')
        self.assertIsNone(p.descriptions_dict[y_2])
        self.assertEquals(p.descriptions_dict[y_3], 'bar')

        self.assertEquals(p.ordered_descriptions[0], 'foo')
        self.assertIsNone(p.ordered_descriptions[1])
        self.assertEquals(p.ordered_descriptions[2], 'bar')

    def test_ode_moment_description_generation_non_existent_key(self):
        """
        Given a key in the specification dict that does not exist, the ODE moment should raise a KeyError
        """

        lhs = sympy.Matrix(['y_1', 'y_2', 'y_3'])
        rhs = sympy.Matrix(['y_1+y_2+c_2', 'y_2+y_3+c_3', 'y_3+c_1'])
        y_1 = sympy.Symbol('y_1')
        y_3 = sympy.Symbol('y_3')
        y_4 = sympy.Symbol('y_4')

        descriptions = {y_1: 'foo', y_3: 'bar', y_4: 'non-existent'}
        self.assertRaises(KeyError, ODEProblem, 'MEA', lhs, rhs, constants=sympy.symbols(['c_1', 'c_2', 'c_3']),
                          description_of_lhs_terms=descriptions)

        descriptions = {'y_1': 'foo', 'y_3': 'bar', 'y_4': 'non-existent'}
        self.assertRaises(KeyError, ODEProblem, 'MEA', lhs, rhs, constants=sympy.symbols(['c_1', 'c_2', 'c_3']),
                          description_of_lhs_terms=descriptions)