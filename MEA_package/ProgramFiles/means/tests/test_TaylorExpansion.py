import unittest

from sympy import Matrix, diff, Symbol, simplify, S
import sympy as sp

from fcount import fcount
from means.approximation.mea.TaylorExpansion import taylor_expansion
from means.approximation.mea.TaylorExpansion import get_factorial_term
from means.approximation.mea.TaylorExpansion import derive_expr_from_counter_entry


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


        nMoments = 3
        nvariables = 2
        variables = ["y_0","y_1"]

        a_strings = ["c_0*y_0*(120-301+y_0+y_1)", "c_1*(301-(y_0+y_1))", "c_2*(301-(y_0+y_1))"]
        z = S(0)

        damat = Matrix(nMoments, 1, lambda i, j : 0)

        damat[0,0] = [[diff(simplify(a_strings[0]), Symbol("y_0")), simplify("c_0*y_0")],
                                [simplify("-c_1"), simplify("-c_1")],
                                [simplify("-c_2"), simplify("-c_2")]]

        damat[1,0] = [[simplify("2*c_0"), simplify("c_0"), simplify("c_0"), z],
                                [z] * 4,
                                [z] * 4]
        damat[2,0] = [[z]*8] * 3

        nreactions = len(a_strings)

        amat = Matrix(nMoments, 1, lambda i, j : simplify(a_strings[i]))

        counter = fcount(nMoments, nvariables)[0]

        #te_result = taylor_expansion(nreactions, nvariables, damat, amat, counter, nMoments)
        te_result = taylor_expansion(variables, amat, counter)

        # hard codding the expected matrix:
        expected_te_mat = Matrix(nreactions, len(counter), lambda i, j : 0)

        expected_te_mat[0,0] = simplify("c_0*y_0*(y_0 + y_1 - 181)")
        expected_te_mat[1,0] = simplify("c_1*(-y_0 - y_1 + 301)")
        expected_te_mat[2,0] = simplify("c_2*(-y_0 - y_1 + 301)")
        expected_te_mat[0, 3],expected_te_mat[0, 5] = (Symbol("c_0"),Symbol("c_0"))

        self.assertEqual(expected_te_mat, te_result)
