import unittest
import sympy
from means.util import sympyhelpers


class TestSympyHelpers(unittest.TestCase):

    def test_substitute_all_on_matrix(self):

        to_substitute = sympy.Matrix(["a*b","c*d","d*e","e*f"])
        pairs = zip(sympy.Matrix(["a","d","e","c","b"]),
                    sympy.Matrix(["z","w","v","x","y"]))

        expected = sympy.Matrix(["z*y","x*w","w*v","v*f"])
        answer = sympyhelpers.substitute_all(to_substitute, pairs)
        self.assertEqual(answer, expected)

    def test_substitute_all_on_expression(self):

        to_substitute = sympy.sympify("a*b + c*d + d*e + e*f")
        pairs = zip(sympy.Matrix(["a","d","e","c","b"]),
                    sympy.Matrix(["z","w","v","x","y"]))
        expected = sympy.sympify("z*y + x*w + w*v + v*f")
        answer = sympyhelpers.substitute_all(to_substitute, pairs)
        self.assertEqual(answer, expected)