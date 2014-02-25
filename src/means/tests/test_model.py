import unittest

import sympy
import numpy as np

from means.model.model import Model
from means.util.sympyhelpers import to_sympy_matrix, assert_sympy_expressions_equal


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
        self.assertEqual(m.species, sympy.symbols(['y_0', 'y_1']))


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
        self.assertEqual(m.species, sympy.symbols(['y_0', 'y_1']))

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
        self.assertEqual(m.species, sympy.symbols(['y_0', 'y_1']))

        # Row
        m = Model(self.SAMPLE_CONSTANTS,
                  sympy.Matrix([['y_0', 'y_1']]),
                  self.SAMPLE_PROPENSITIES,
                  self.SAMPLE_STOICHIOMETRY_MATRIX)
        self.assertEqual(m.species, sympy.symbols(['y_0', 'y_1']))

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
        answer = to_sympy_matrix([['c_0*y_0*(y_0 + y_1 - 181)'],
                               ['c_1*(-y_0 - y_1 + 301)'],
                               ['c_2*(-y_0 - y_1 + 301)']])
        # Column
        m = Model(self.SAMPLE_CONSTANTS,
                  self.SAMPLE_VARIABLES,
                  to_sympy_matrix(['c_0*y_0*(y_0 + y_1 - 181)',
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
        answer = sympy.Matrix([[-1, 1, 0], [0, 0, 1]])

        # Column
        m = Model(self.SAMPLE_CONSTANTS,
                  self.SAMPLE_VARIABLES,
                  self.SAMPLE_PROPENSITIES,
                  sympy.Matrix([[-1, 1, 0], [0, 0, 1]]))
        assert_sympy_expressions_equal(m.stoichiometry_matrix, answer)

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

class TestModelEquality(unittest.TestCase):

    def test_equal_models_are_equal(self):
        """
        Given two equal models, they should return true on == comparison.
        """
        model_a = Model(species=['a', 'b'], constants=['x', 'y'], stoichiometry_matrix=[[2,3], [15,-1]],
                        propensities=['a+b+x+y', 'a-b+x-y'])
        model_b = Model(species=['a', 'b'], constants=['x', 'y'], stoichiometry_matrix=[[2,3], [15,-1]],
                        propensities=['a+b+x+y', 'a-b+x-y'])

        self.assertEqual(model_a, model_b)

    def test_models_with_different_species_are_unequal(self):
        """
        Given models that differ by species names only, they should not be equal
        """
        model_a = Model(species=['a', 'b'], constants=['x', 'y'], stoichiometry_matrix=[[2,3], [15,-1]],
                        propensities=['a+b+x+y', 'a-b+x-y'])
        model_b = Model(species=['c', 'd'], constants=['x', 'y'], stoichiometry_matrix=[[2,3], [15,-1]],
                        propensities=['c+d+x+y', 'c-d+x-y'])
        model_c = Model(species=['c', 'b'], constants=['x', 'y'], stoichiometry_matrix=[[2,3], [15,-1]],
                        propensities=['c+b+x+y', 'c-b+x-y'])

        self.assertNotEqual(model_a, model_b)
        self.assertNotEqual(model_a, model_c)
        self.assertNotEqual(model_b, model_c)

    def test_models_with_different_constants_are_unequal(self):
        """
        Given two models, that differ by their constants only, they should return False on == comparison.
        """
        model_a = Model(species=['a', 'b'], constants=['x', 'y'], stoichiometry_matrix=[[2,3], [15,-1]],
                        propensities=['a+b+x+y', 'a-b+x-y'])
        model_b = Model(species=['a', 'b'], constants=['z', 'q'], stoichiometry_matrix=[[2,3], [15,-1]],
                        propensities=['a+b+z+q', 'a-b+z-q'])
        self.assertNotEqual(model_a, model_b)

    def test_models_with_different_stoichiometry_matrices_are_unequal(self):
        """
        Given two models that differ only by stoichiometry_matrices, they should be returned as unequal.
        """
        model_a = Model(species=['a', 'b'], constants=['x', 'y'], stoichiometry_matrix=[[2,3], [15,-1]],
                        propensities=['a+b+x+y', 'a-b+x-y'])
        model_b = Model(species=['a', 'b'], constants=['x', 'y'], stoichiometry_matrix=[[1,2], [3,-4]],
                        propensities=['a+b+x+y', 'a-b+x-y'])

        self.assertNotEqual(model_a, model_b)

    def test_models_with_different_propensities_are_unequal(self):
        """
        Given two models, that differ by their propensities only, the == comparison should return false
        """
        model_a = Model(species=['a', 'b'], constants=['x', 'y'], stoichiometry_matrix=[[2,3], [15,-1]],
                        propensities=['a+b+x+y', 'a-b+x-y'])
        model_b = Model(species=['a', 'b'], constants=['x', 'y'], stoichiometry_matrix=[[2,3], [15,-1]],
                        propensities=['a+b+x+y', 'a-b+x'])

        self.assertNotEqual(model_a, model_b)

    def test_models_with_equivalent_propensities_are_equal(self):
        """
        Given two equal models, that have different, but equivalent propensities,
        the == comparison should return true
        """
        model_a = Model(species=['a', 'b'], constants=['x', 'y'], stoichiometry_matrix=[[2,3], [15,-1]],
                        propensities=['a+b+x+y', 'a-b+x-y'])
        model_b = Model(species=['a', 'b'], constants=['x', 'y'], stoichiometry_matrix=[[2,3], [15,-1]],
                        propensities=['a+b+x+y', 'a-b+x-y +2*x +x - (1/3) * x * 9'])

        self.assertEqual(model_a, model_b)