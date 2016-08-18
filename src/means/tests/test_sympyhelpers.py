from __future__ import absolute_import, print_function

import unittest

import sympy

from means.util.sympyhelpers import to_sympy_matrix, assert_sympy_expressions_equal, sympy_expressions_equal
from means.util.sympyhelpers import substitute_all

class TestSympyHelpers(unittest.TestCase):

    def test_substitute_all_on_matrix(self):

        to_substitute = to_sympy_matrix(["a*b","c*d","d*e","e*f"])
        pairs = zip(to_sympy_matrix(["a","d","e","c","b"]),
                    to_sympy_matrix(["z","w","v","x","y"]))

        expected = sympy.Matrix(["z*y","x*w","w*v","v*f"])
        answer = substitute_all(to_substitute, pairs)
        self.assertEqual(answer, expected)

    def test_substitute_all_on_expression(self):

        to_substitute = sympy.sympify("a*b + c*d + d*e + e*f")
        pairs = zip(to_sympy_matrix(["a","d","e","c","b"]),
                    to_sympy_matrix(["z","w","v","x","y"]))
        expected = sympy.sympify("z*y + x*w + w*v + v*f")
        answer = substitute_all(to_substitute, pairs)
        self.assertEqual(answer, expected)



class TestSympyExpressionsEqual(unittest.TestCase):

    def test_matrices(self):
        """
        Given two equal matrices, `sympy_expressions_equal` should correctly identify them as equivalent.
        Given two different matrices, it should correctly call them different
        """

        m1 = sympy.Matrix([[sympy.sympify('x'), 1, 2], [3, 4, 5]])
        m2 = sympy.Matrix([[sympy.sympify('x'), 1, 2], [sympy.sympify('3 + x - x'), 4, 5]])
        m3 = sympy.Matrix([[sympy.sympify('x'), 1, 2], [sympy.sympify('3 + x - y'), 4, 5]])

        self.assertTrue(sympy_expressions_equal(m1, m2))
        self.assertFalse(sympy_expressions_equal(m1, m3))
        self.assertFalse(sympy_expressions_equal(m2, m3))

    def test_expressions(self):
        """
        Given two equivalent expressions, `sympy_expresions_equal` should correctly call them equal.
        Given two different expressions, the function should say they are not equal.
        """

        e1 = sympy.sympify('x+y+z')
        e2 = sympy.sympify('0.5 * (2*x + 2*y + 2*z)')
        e3 = sympy.sympify('x+y+z - y')

        self.assertTrue(sympy_expressions_equal(e1, e2))
        self.assertFalse(sympy_expressions_equal(e1, e3))
        self.assertFalse(sympy_expressions_equal(e2, e3))

class TestToSympyMatrix(unittest.TestCase):


    def test_creation_from_matrix_returns_itself(self):
        """
        Given a `sympy.Matrix`, `to_sympy_matrix` should return the said matrix.
        """

        m = sympy.Matrix([[1, 2, 3], [4, 5, 6]])
        assert_sympy_expressions_equal(m, to_sympy_matrix(m))

    def test_creation_from_list_of_integers_returns_matrix(self):
        """
        Given a list of integers, to_sympy_matrix should be able to convert it to a matrix of these integers
        :return:
        """

        m = sympy.Matrix([[1, 2, 3], [4, 5, 6]])
        m_as_list = [[1, 2, 3], [4, 5, 6]]

        assert_sympy_expressions_equal(m, to_sympy_matrix(m_as_list))

    def test_creation_from_list_of_strings_returns_matrix(self):
        """
        Given a list of strings, to_sympy_matrix should be able to convert them into a matrix of expressions.
        """

        m = sympy.Matrix([[sympy.sympify('x+y+3'), sympy.sympify('x+3')],
                          [sympy.sympify('y-x'), sympy.sympify('x+y+166')]])

        m_as_string = [['x+y+3', 'x+3'], ['y-x', 'x+y+166']]
        matrix = to_sympy_matrix(m_as_string)
        assert_sympy_expressions_equal(m, matrix)

    def test_creation_of_column_matrix_from_list_of_strings(self):
        """
        Given a list of strings, to_sympy_matrix should be able to convert them into a column matrix of expresions
        """
        m = sympy.Matrix([sympy.sympify('x+y+3'), sympy.sympify('x+3'), sympy.sympify('y-x'),
                           sympy.sympify('x+y+166')])

        m_as_string = ['x+y+3', 'x+3', 'y-x', 'x+y+166']

        matrix = to_sympy_matrix(m_as_string)
        assert_sympy_expressions_equal(m, matrix)
