import unittest
import sympy
import numpy as np
from model import Model

class TestModelInitialisation(unittest.TestCase):

    SAMPLE_CONSTANTS = sympy.symbols(['c_0', 'c_1', 'c_2'])
    SAMPLE_VARIABLES = sympy.symbols(['y_0', 'y_1'])
    SAMPLE_PROPENSITIES = sympy.Matrix([['c_0*y_0*(y_0 + y_1 - 181)'],
                                        ['c_1*(-y_0 - y_1 + 301)'],
                                        ['c_2*(-y_0 - y_1 + 301)']])

    SAMPLE_STOICHIOMETRY_MATRIX = sympy.Matrix([[-1, 1, 0], [0, 0, 1]])

    #-- Constants -----------------------------------------------------

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

    #-- Variables ---------------------------------------------------

    def test_initialisation_of_variables_as_list_of_strings(self):
        """
        The model constructor should accept variables as
        a list of strings, i.e. ['y_0', 'y_1']
        It should return them as sympy.symbols(['y_0', 'y_1'])
        :return:
        """

        m = Model(self.SAMPLE_CONSTANTS,
                  ['y_0', 'y_1'],
                  self.SAMPLE_PROPENSITIES,
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.variables, sympy.symbols(['y_0', 'y_1']))


    def test_initialisation_of_variables_as_list_of_sympy_symbols(self):
        """
        The model constructor should accept variables as
        a list of sympy symbols i.e. sympy.symbols(['y_0', 'y_1'])
        It should return them as sympy.symbols(['y_0', 'y_1'])
        :return:
        """

        m = Model(self.SAMPLE_CONSTANTS,
                  sympy.symbols(['y_0', 'y_1']),
                  self.SAMPLE_PROPENSITIES,
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.variables, sympy.symbols(['y_0', 'y_1']))

    def test_initialisation_of_variables_as_list_of_sympy_matrix(self):
        """
        The model constructor should accept variables as
        sympy.Matrix i.e. sympy.Matrix(['y_0', 'y_1'])
        It should return them as sympy.symbols(['y_0', 'y_1'])
        :return:
        """
        # Column
        m = Model(self.SAMPLE_CONSTANTS,
                  sympy.Matrix(['y_0', 'y_1']),
                  self.SAMPLE_PROPENSITIES,
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.variables, sympy.symbols(['y_0', 'y_1']))

        # Row
        m = Model(self.SAMPLE_CONSTANTS,
                  sympy.Matrix([['y_0', 'y_1']]),
                  self.SAMPLE_PROPENSITIES,
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.variables, sympy.symbols(['y_0', 'y_1']))

    #-- Propensity matrix -------------------------------------------------

    def test_initialisation_of_propensities_as_list_of_strings(self):
        """
        The model constructor should accept propensities as list of strings
        e.g. ['y_0+y_1', 'y_1+y_2']
        and return them as sympy (column) Matrix of equations
        i.e. sympy.Matrix(['y_0+y_1', 'y_1+y_2'])
        """
        answer = sympy.Matrix([['c_0*y_0*(y_0 + y_1 - 181)'],
                               ['c_1*(-y_0 - y_1 + 301)'],
                               ['c_2*(-y_0 - y_1 + 301)']])
        # List
        m = Model(self.SAMPLE_CONSTANTS,
                  self.SAMPLE_VARIABLES,
                  ['c_0*y_0*(y_0 + y_1 - 181)',
                   'c_1*(-y_0 - y_1 + 301)',
                   'c_2*(-y_0 - y_1 + 301)'],
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.propensities, answer)

        # List of lists
        m = Model(self.SAMPLE_CONSTANTS,
                  self.SAMPLE_VARIABLES,
                  [['c_0*y_0*(y_0 + y_1 - 181)'],
                   ['c_1*(-y_0 - y_1 + 301)'],
                   ['c_2*(-y_0 - y_1 + 301)']],
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.propensities, answer)

        # Double list
        m = Model(self.SAMPLE_CONSTANTS,
                  self.SAMPLE_VARIABLES,
                  [['c_0*y_0*(y_0 + y_1 - 181)',
                    'c_1*(-y_0 - y_1 + 301)',
                    'c_2*(-y_0 - y_1 + 301)']],
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.propensities, answer)

    def test_initialisation_of_propensities_as_matrix(self):
        """
        The model constructor should accept propensities as a sympy matrix
        e.g. sympy.Matrix(['y_0+y_1', 'y_1+y_2'])
        and return them as sympy (column) Matrix of equations
        i.e. sympy.Matrix(['y_0+y_1', 'y_1+y_2'])
        """
        answer = sympy.Matrix([['c_0*y_0*(y_0 + y_1 - 181)'],
                               ['c_1*(-y_0 - y_1 + 301)'],
                               ['c_2*(-y_0 - y_1 + 301)']])
        # Column
        m = Model(self.SAMPLE_CONSTANTS,
                  self.SAMPLE_VARIABLES,
                  sympy.Matrix(['c_0*y_0*(y_0 + y_1 - 181)',
                               'c_1*(-y_0 - y_1 + 301)',
                               'c_2*(-y_0 - y_1 + 301)']),
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.propensities, answer)

        # Row matrix
        m = Model(self.SAMPLE_CONSTANTS,
                  self.SAMPLE_VARIABLES,
                  [['c_0*y_0*(y_0 + y_1 - 181)',
                   'c_1*(-y_0 - y_1 + 301)',
                   'c_2*(-y_0 - y_1 + 301)']],
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.propensities, answer)

    def test_initialisation_of_propensities_as_list_of_sympy_formulae(self):
        """
        The model constructor should accept propensities as list of strings
        e.g. map(sympy.sympify, ['y_0+y_1', 'y_1+y_2'])
        and return them as sympy (column) Matrix of equations
        i.e. sympy.Matrix(['y_0+y_1', 'y_1+y_2'])
        """
        answer = sympy.Matrix([['c_0*y_0*(y_0 + y_1 - 181)'],
                               ['c_1*(-y_0 - y_1 + 301)'],
                               ['c_2*(-y_0 - y_1 + 301)']])
        # List
        m = Model(self.SAMPLE_CONSTANTS,
                  self.SAMPLE_VARIABLES,
                  map(sympy.sympify, ['c_0*y_0*(y_0 + y_1 - 181)',
                                      'c_1*(-y_0 - y_1 + 301)',
                                      'c_2*(-y_0 - y_1 + 301)']),
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.propensities, answer)

        # Double list
        m = Model(self.SAMPLE_CONSTANTS,
                  self.SAMPLE_VARIABLES,
                  [map(sympy.sympify, ['c_0*y_0*(y_0 + y_1 - 181)',
                                       'c_1*(-y_0 - y_1 + 301)',
                                       'c_2*(-y_0 - y_1 + 301)'])],
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.propensities, answer)

    #-- Stoichiometry matrix -----------------------------------------------

    def test_initialisation_of_stoichiometry_matrix_as_list(self):
        """
        The model constructor should accept stoichiometry_matrix as list
        e.g. [[-1, 1, 0], [0, 0, 1]]
        and return them as sympy Matrix
        e.g. sympy.Matrix([[-1, 1, 0], [0, 0, 1]])
        """
        answer = sympy.Matrix([[-1, 1, 0], [0, 0, 1]])
        # List
        m = Model(self.SAMPLE_CONSTANTS,
                  self.SAMPLE_VARIABLES,
                  self.SAMPLE_PROPENSITIES,
                  [[-1, 1, 0], [0, 0, 1]])
        self.assertEqual(m.stoichiometry_matrix, answer)

    def test_initialisation_of_stoichiometry_matrix_as_matrix(self):
        """
        The model constructor should accept stoichiometry_matrix as a sympy matrix
        e.g. sympy.Matrix([[-1, 1, 0], [0, 0, 1]])
        and return them as sympy Matrix
        e.g. sympy.Matrix([[-1, 1, 0], [0, 0, 1]])
        """
        answer = sympy.Matrix([[-1, 1, 0], [0, 0, 1]])

        # Column
        m = Model(self.SAMPLE_CONSTANTS,
                  self.SAMPLE_VARIABLES,
                  self.SAMPLE_PROPENSITIES,
                  sympy.Matrix([[-1, 1, 0], [0, 0, 1]]))
        self.assertEqual(m.stoichiometry_matrix, answer)

    def test_initialisation_of_stoichiometry_matrix_as_numpy_array(self):
        """
        The model constructor should accept stoichiometry_matrix as a numpy matrix
        e.g. np.array(['y_0+y_1', 'y_1+y_2'])
        and return them as sympy Matrix
        e.g. sympy.Matrix([[-1, 1, 0], [0, 0, 1]])
        """
        answer = np.array([[-1, 1, 0], [0, 0, 1]])

        # Column
        m = Model(self.SAMPLE_CONSTANTS,
                  self.SAMPLE_VARIABLES,
                  self.SAMPLE_PROPENSITIES,
                  sympy.Matrix([[-1, 1, 0], [0, 0, 1]]))
        self.assertEqual(m.stoichiometry_matrix, answer)

    #-- Validation ----------------------------------------------------------

    def test_model_validates_the_size_of_stoichiometry_and_propensity_matrices(self):
        """
        Given a stoichiometry matrix and propensity matrix, the model should raise a value error
        if the number of columns in stoichiometry matrix
        is not equal to the number of rows in propensity matrix
        :return:
        """
        self.assertRaises(ValueError, Model,
                          self.SAMPLE_CONSTANTS, self.SAMPLE_VARIABLES,
                          self.SAMPLE_PROPENSITIES, sympy.Matrix([[1, 1], [0, 0]]))

    def test_model_validates_the_size_of_stoichiometry_matrix_and_number_of_variables(self):
        """
        Given a stoichiometry matrix and variables list, the model should raise a value error
        if the number of rows in stoichiometry matrix is not equal to number of variables
        :return:
        """
        self.assertRaises(ValueError, Model,
                          self.SAMPLE_CONSTANTS, self.SAMPLE_VARIABLES,
                          self.SAMPLE_PROPENSITIES, sympy.Matrix([[1, 1, 1]]))

    def test_model_validates_that_propensities_is_a_vector(self):
        """
        Given a matrix (not a vector) for propensity matrix,
        the creation model should fail with ValueError
        :return:
        """
        self.assertRaises(ValueError, Model,
                          self.SAMPLE_CONSTANTS,
                          self.SAMPLE_VARIABLES,
                          sympy.Matrix([[1, 2], [3, 4], [5, 6]]),
                          self.SAMPLE_STOICHIOMETRY_MATRIX)
