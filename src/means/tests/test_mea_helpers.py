from __future__ import absolute_import, print_function

import unittest

import sympy as sp

from means.approximation.mea.mea_helpers import get_one_over_n_factorial, derive_expr_from_counter_entry
from means.util.sympyhelpers import assert_sympy_expressions_equal


class TaylorExpansionTestCase(unittest.TestCase):
    def test_derive_expr_from_counter_entry(self):
        """
        Given the tuples of integers a, b and c
        Then, the "composite derivatives" should be exactly "a_result", "b_result" and "c_result", respectively

        :return:
        """

        expr = sp.simplify("c_0*y_0*(y_0 + y_1 - 181)/(y_2+c_1*y_1)")

        vars = sp.simplify(["y_0", "y_1", "y_2"])

        count_entr_a = (0, 1, 3)
        count_entr_b = (1, 1, 0)
        count_entr_c = (0, 0, 0)

        a_result = derive_expr_from_counter_entry(expr, vars, count_entr_a)
        b_result = derive_expr_from_counter_entry(expr, vars, count_entr_b)
        c_result = derive_expr_from_counter_entry(expr, vars, count_entr_c)

        a_expected = sp.diff(sp.diff(expr, "y_2", 3), "y_1")
        b_expected = sp.diff(sp.diff(expr, "y_0"), "y_1")
        c_expected = expr

        assert_sympy_expressions_equal(a_expected, a_result)
        assert_sympy_expressions_equal(b_expected, b_result)
        assert_sympy_expressions_equal(c_expected, c_result)


    def test_get_factorial_term(self):
        """
        Given the tuples of integers a and b,
        Then, the "factorial term" should be exactly "a_result" and "b_result", respectively

        :return:
        """

        a = (2, 3, 4)
        b = (0, 1, 6)
        a_expected = sp.S(1) / (sp.factorial(2) * sp.factorial(3) * sp.factorial(4))
        b_expected = sp.S(1) / (sp.factorial(6))
        a_result = get_one_over_n_factorial(a)
        b_result = get_one_over_n_factorial(b)
        self.assertEqual(a_expected, a_result)
        self.assertEqual(b_expected, b_result)
