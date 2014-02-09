import unittest

import sympy as sp

from means.approximation.mea.TaylorExpansion import generate_dmu_over_dt
from means.approximation.mea.TaylorExpansion import get_factorial_term
from means.approximation.mea.TaylorExpansion import derive_expr_from_counter_entry
from means.approximation.ode_problem import Moment
from means.approximation.mea.moment_expansion_approximation import MomentExpansionApproximation

class TaylorExpansionTestCase(unittest.TestCase):

    def test_derive_expr_from_counter_entry(self):

        """
        Given the tuples of integers a, b and c
        Then, the "composite derivatives" should be exactly "a_result", "b_result" and "c_result", respectively

        :return:
        """


        expr = sp.simplify("c_0*y_0*(y_0 + y_1 - 181)/(y_2+c_1*y_1)")

        vars = sp.simplify(["y_0", "y_1", "y_2"])

        count_entr_a = (0,1,3)
        count_entr_b = (1,1,0)
        count_entr_c = (0,0,0)

        a_result = derive_expr_from_counter_entry(expr, vars, count_entr_a)
        b_result = derive_expr_from_counter_entry(expr, vars, count_entr_b)
        c_result = derive_expr_from_counter_entry(expr, vars, count_entr_c)

        a_expected = sp.diff(sp.diff(expr,"y_2",3), "y_1")
        b_expected = sp.diff(sp.diff(expr,"y_0"), "y_1")
        c_expected = expr

        self.assertEqual(a_expected, a_result)
        self.assertEqual(b_expected, b_result)
        self.assertEqual(c_expected, c_result)



    def test_get_factorial_term(self):

        """
        Given the tuples of integers a and b,
        Then, the "factorial term" should be exactly "a_result" and "b_result", respectively

        :return:
        """

        a = (2,3,4)
        b = (0,1,6)
        a_expected = sp.S(1)/(sp.factorial(2) * sp.factorial(3) * sp.factorial(4))
        b_expected =  sp.S(1)/(sp.factorial(6))
        a_result = get_factorial_term(a)
        b_result = get_factorial_term(b)
        self.assertEqual(a_expected, a_result)
        self.assertEqual(b_expected, b_result)

    def test_TaylorExpansion(self):
        """
        Given the number of moments is 3, the number of species is 2,
        Given the propensities of the 3 reactions in `a_strings`,
        And Given the combination of derivative order in counter,
        Then results of `TaylorExpansion()` should produce a matrix exactly equal to
        exactly equal to the the expected one (`expected_te_matrix`).

        :return:
        """

        mea = MomentExpansionApproximation(None,3)
        species = sp.Matrix(["a","b","c"])
        propensities = sp.Matrix(["a*2 +w * b**3","b - a*x /c","c + a*b /32"])
        stoichiometry_matrix = sp.Matrix([
                        [1, 0, 1],
                        [-1, -1, 0],
                        [0, 1, -1]
                                      ])

        counter = [
            Moment([0,0,2],sp.Symbol("q1")),
            Moment([0,2,0],sp.Symbol("q2")),
            Moment([0,0,2],sp.Symbol("q3")),
            Moment([2,0,0],sp.Symbol("q4")),
            Moment([1,1,0],sp.Symbol("q5")),
            Moment([0,1,1],sp.Symbol("q6")),
            Moment([1,0,1],sp.Symbol("q7"))]

        result =  generate_dmu_over_dt(species,propensities,counter, stoichiometry_matrix)
        expected = stoichiometry_matrix * sp.Matrix([  ["        0", "3*b*w",         "0", "0",    "0", "0",     "0"],
                                ["-a*x/c**3",     "0", "-a*x/c**3", "0",    "0", "0", "x/c**2"],
                                [        "0",     "0",         "0", "0", "1/32", "0",      "0"]])

        self.assertEqual(result, expected)
