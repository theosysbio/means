import unittest
import sympy
from formatmodel import Model

class TestModelInitialisation(unittest.TestCase):

    SAMPLE_CONSTANTS = sympy.symbols(['c_0', 'c_1', 'c_2'])
    SAMPLE_VARIABLES = sympy.symbols(['y_0', 'y_1'])
    SAMPLE_PROPENSITIES = sympy.Matrix([['c_0*y_0*(y_0 + y_1 - 181)'],
                                        ['c_1*(-y_0 - y_1 + 301)'],
                                        ['c_2*(-y_0 - y_1 + 301)']])

    SAMPLE_STOICHIOMETRY_MATRIX = sympy.Matrix([[-1, 1, 0], [0, 0, 1]])

    def test_initialisation_of_constants_as_list_of_strings(self):
        """
        The model constructor should accept constants as
        a list of strings, i.e. ['c_0', 'c_1', 'c_2']
        It should return them as sympy.symbols(['c_0', 'c_1', 'c_2'])
        :return:
        """

        m = Model(['c_0', 'c_1', 'c_2'],
                  self.SAMPLE_VARIABLES,
                  self.SAMPLE_PROPENSITIES,
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.constants, sympy.symbols(['c_0', 'c_1', 'c_2']))

    def test_initialisation_of_constants_as_list_of_sympy_symbols(self):
        """
        The model constructor should accept constants as
        a list of sympy symbols i.e. sympy.symbols(['c_0', 'c_1', 'c_2'])
        It should return them as sympy.symbols(['c_0', 'c_1', 'c_2'])
        :return:
        """

        m = Model(sympy.symbols(['c_0', 'c_1', 'c_2']),
                  self.SAMPLE_VARIABLES,
                  self.SAMPLE_PROPENSITIES,
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.constants, sympy.symbols(['c_0', 'c_1', 'c_2']))

    def test_initialisation_of_constants_as_list_of_sympy_matrix(self):
        """
        The model constructor should accept constants as
        sympy.Matrix i.e. sympy.Matrix(['c_0', 'c_1', 'c_2'])
        It should return them as sympy.symbols(['c_0', 'c_1', 'c_2'])
        :return:
        """
        # Column
        m = Model(sympy.Matrix(['c_0', 'c_1', 'c_2']),
                  self.SAMPLE_VARIABLES,
                  self.SAMPLE_PROPENSITIES,
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.constants, sympy.symbols(['c_0', 'c_1', 'c_2']))

        # Row
        m = Model(sympy.Matrix([['c_0', 'c_1', 'c_2']]),
                  self.SAMPLE_VARIABLES,
                  self.SAMPLE_PROPENSITIES,
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.constants, sympy.symbols(['c_0', 'c_1', 'c_2']))





