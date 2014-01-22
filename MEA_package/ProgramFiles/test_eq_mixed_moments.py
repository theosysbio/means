import unittest
import sympy
from eq_mixedmoments import eq_mixedmoments


class AbstractTestEqMixedMoments(unittest.TestCase):
    def get_instance_values(self):
        raise NotImplementedError

    def setUp(self):
        constants, species, number_of_moments, stoichiometry_matrix, propensities, counter = self.get_instance_values()

        self.CONSTANTS = constants
        self.SPECIES = species
        self.NUMBER_OF_MOMENTS = number_of_moments
        self.STOICHIOMETRY_MATRIX = stoichiometry_matrix
        self.PROPENSITIES = propensities
        self.COUNTER = counter

    def fetch_answer(self, kvec, ekcounter, da_dt):
        """
        Returns what `eq_mixedmoments` function returns for the specified counter, kvec, ekcounter and da_dt parameters
        :param kvec:
        :param ekcounter:
        :param da_dt:
        :return:
        """

        # These are the arguments as it is called for model MM
        KWARGS = {'nreactions': len(self.PROPENSITIES),
                  'nvariables': len(self.SPECIES),
                  'nMoments': self.NUMBER_OF_MOMENTS,
                  'amat': self.PROPENSITIES,
                  'counter': self.COUNTER,
                  'S': self.STOICHIOMETRY_MATRIX,
                  'ymat': self.SPECIES,
                  'nDerivatives': self.NUMBER_OF_MOMENTS,
                  'kvec': kvec,
                  'ekcounter': ekcounter,
                  # This one must be a column vector for the code to work!
                  'dAdt': da_dt
        }

        answer = eq_mixedmoments(**KWARGS)
        return answer

class TestEqMixedMoments_Under_MM_model(AbstractTestEqMixedMoments):
    """
    Tests the EqMixedMoments function under MM model as it is specified
    """

    def get_instance_values(self):
        constants = sympy.Matrix([sympy.var('c_0'), sympy.var('c_1'), sympy.var('c_2')])
        species = sympy.Matrix([sympy.var('y_0'), sympy.var('y_1')])
        number_of_moments = 2
        stoichiometry_matrix = sympy.Matrix([[-1, 1, 0], [0, 0, 1]])
        propensities = sympy.Matrix([constants[0] * species[0] * (species[0] + species[1] - 181),
                                     constants[1] * (-1 * species[0] - species[1] + 301),
                                     constants[2] * (-1 * species[0] - species[1] + 301)])
        counter = [[0, 0], [0, 2], [1, 1], [2, 0]]

        return constants, species, number_of_moments, stoichiometry_matrix, propensities, counter


    def test_first_iteration_conditions(self):
        """
        Given that the initial conditions are given as they are given for eq_mixedmoments the first time it is called
        The program should return the same results as it does without us touching it
        :return:
        """
        CONSTANTS, SPECIES = self.CONSTANTS, self.SPECIES  # To simplify the uses of this below

        da_dt = sympy.Matrix([[CONSTANTS[2] * (-1 * SPECIES[0] - SPECIES[1] + 301), 0, 0, 0]])
        answer = self.fetch_answer(kvec=[0, 1],
                                   ekcounter=[[0, 1]],
                                   da_dt=da_dt)

        correct_answer = da_dt  # This test just returns dAdt even though dAdt is not used like that in the code ... weird
        self.assertEqual(answer, correct_answer)


    def test_last_iteration_conditions(self):
        """
        Given that the initial conditions are given as they are given for eq_mixedmoments the first time it is called
        The program should return the same results as it does without us touching it
        :return:
        """
        CONSTANTS, SPECIES = self.CONSTANTS, self.SPECIES  # To simplify the uses of this below

        da_dt = sympy.Matrix([[0, 0, 0, 0]])
        answer = self.fetch_answer(kvec=[2, 0],
                                   ekcounter=[[1, 0], [2, 0]],
                                   da_dt=sympy.Matrix([[0, 0, 0, 0]])
        )

        correct_answer = sympy.Matrix([[sympy.sympify(
            '-2*c_0*y_0**2*(y_0 + y_1 - 181) + c_0*y_0*(y_0 + y_1 - 181) + 2*c_1*y_0*(-y_0 - y_1 + 301) + c_1*(-y_0 - y_1 + 301)'),
                                       0,
                                       sympy.sympify('-4*c_0*y_0 + c_0 - 2*c_1'),
                                       sympy.sympify('-4*c_0*y_0 - 2*c_0*(y_0 + y_1 - 181) + c_0 - 2*c_1')
        ]])
        self.assertEqual(answer, correct_answer)

class TestEqMixedMoments_Under_p53(AbstractTestEqMixedMoments):
    """
    Tests the EqMixedMoments function under P53 model as it is specified
    """

    def get_instance_values(self):
        constants = sympy.Matrix([sympy.var('c_0'), sympy.var('c_1'), sympy.var('c_2'), sympy.var('c_3'), sympy.var('c_4'), sympy.var('c_5'), sympy.var('c_6')])
        species = sympy.Matrix([sympy.var('y_0'), sympy.var('y_1'), sympy.var('y_2')])
        number_of_moments = 2
        stoichiometry_matrix = sympy.Matrix([[1, -1, -1, 0, 0, 0],
                                             [0, 0, 0, 1, -1, 0],
                                             [0, 0, 0, 0, 1, -1]])
        propensities = sympy.Matrix([constants[0],
                                     constants[1] * species[0],
                                     constants[2] * species[0] * species[2] / (constants[6] + species[0]),
                                     constants[3] * species[0],
                                     constants[4] * species[1],
                                     constants[5] * species[2]])
        counter = [[0, 0, 0], [0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]

        return constants, species, number_of_moments, stoichiometry_matrix, propensities, counter


    def test_random_case(self):
        """
        For an (actual) usage case of this program picked at random, the outputs should match up
        :return:
        """
        answer = self.fetch_answer(kvec=[1, 0, 1],
                                   ekcounter=[[0, 0, 1], [1, 0, 0], [1, 0, 1]],
                                   da_dt=sympy.Matrix([[0, 0, 0, 0, 0, 0, 0]])
        )

        correct_answer = sympy.Matrix([
            map(sympy.sympify, [
                'c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) + c_4*y_0*y_1 - c_5*y_0*y_2',
                '-c_2*y_0/(c_6 + y_0)',
                0,
                0,
                '-c_1 + 2*c_2*y_0*y_2/(c_6 + y_0)**2 - 2*c_2*y_2/(c_6 + y_0) - c_5',
                'c_4',
                '-c_2*y_0*y_2**2/(c_6 + y_0)**3 + c_2*y_2**2/(c_6 + y_0)**2'
               ])])

        self.assertEqual(answer, correct_answer)


    def test_nonzero_da_dt_case(self):
        """
        For a case that has a da_dt matrix that is not zero, the output should
        """
        da_dt = sympy.Matrix([map(sympy.sympify,
                                  ['c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)',
                                   0,
                                   0,
                                   0,
                                   'c_2*y_0/(c_6 + y_0)**2 - c_2/(c_6 + y_0)',
                                   0,
                                   '-c_2*y_0*y_2/(c_6 + y_0)**3 + c_2*y_2/(c_6 + y_0)**2'
                                  ])])

        answer = self.fetch_answer(kvec=[0, 1, 0],
                                   ekcounter=[[0, 1, 0]],
                                   da_dt=da_dt)

        correct_answer = sympy.Matrix([
            map(sympy.sympify, [
               'c_3*y_0 - c_4*y_1',
               0,
               0,
               0,
               0,
               0,
               0
               ])])

        self.assertEqual(answer, correct_answer)

class TestEqMixedMoments_Under_dimer(AbstractTestEqMixedMoments):
    """
    Tests the EqMixedMoments function under dimer model as it is specified
    """

    def get_instance_values(self):
        constants = sympy.Matrix([sympy.var('c_0'), sympy.var('c_1'), sympy.var('c_2')])
        species = sympy.Matrix([sympy.var('y_0')])
        number_of_moments = 2
        stoichiometry_matrix = sympy.Matrix([[-2, 2]])
        propensities = sympy.Matrix([constants[0] * species[0] * (species[0] - 1),
                                     constants[1] * (0.5 * constants[2] - 0.5 * species[0])])
        counter = [[0], [2]]

        return constants, species, number_of_moments, stoichiometry_matrix, propensities, counter


    def test_random_case(self):
        """
        For an (actual) usage case of this program picked at random, the outputs should match up
        :return:
        """
        answer = self.fetch_answer(kvec=[2],
                                   ekcounter=[[1], [2]],
                                   da_dt=sympy.Matrix([[0, 0]])
        )

        correct_answer = sympy.Matrix([
            map(sympy.sympify, [
                '-4*c_0*y_0**2*(y_0 - 1) + 4*c_0*y_0*(y_0 - 1) + 4*c_1*y_0*(0.5*c_2 - 0.5*y_0) + 4*c_1*(0.5*c_2 - 0.5*y_0)',
                '-8*c_0*y_0 - 4*c_0*(y_0 - 1) + 4*c_0 - 2.0*c_1'
               ])])

        self.assertEqual(answer, correct_answer)


    def test_da_dt_non_zero_ase(self):
        """
        For an (actual) usage case of this program that has da_dt non zero, the outputs should match up
        :return:
        """
        da_dt = sympy.Matrix([map(sympy.sympify, ['-2*c_0*y_0*(y_0 - 1) + 2*c_1*(0.5*c_2 - 0.5*y_0)',
                                                                           '-2*c_0'])])

        answer = self.fetch_answer(kvec=[1],
                                   ekcounter=[[1]],
                                   da_dt=da_dt)

        correct_answer = da_dt
        self.assertEqual(answer, correct_answer)

class TestEqMixedMoments_Under_hes1(AbstractTestEqMixedMoments):
    """
    Tests the EqMixedMoments function under dimer model as it is specified
    """

    def get_instance_values(self):
        constants = sympy.Matrix([sympy.var('c_0'), sympy.var('c_1'), sympy.var('c_2'), sympy.var('c_3')])
        species = sympy.Matrix([sympy.var('y_0'), sympy.var('y_1'), sympy.var('y_2')])
        number_of_moments = 2
        stoichiometry_matrix = sympy.Matrix([[-1,  0,  0,  0, 0, 1],
                                             [0, -1,  0, -1, 1, 0],
                                             [ 0,  0, -1,  1, 0, 0]])
        propensities = sympy.Matrix([0.03*species[0],
                                     0.03*species[1],
                                     0.03*species[2],
                                     constants[3] * species[1],
                                     constants[2] * species[0],
                                     1.0/(1+species[2]**2/constants[0]**2)])
        counter = [[0, 0, 0], [0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]

        return constants, species, number_of_moments, stoichiometry_matrix, propensities, counter


    def test_random_case(self):
        """
        For an (actual) usage case of this program picked at random, the outputs should match up
        :return:
        """
        answer = self.fetch_answer(kvec=[2, 0, 0],
                                   ekcounter=[[1, 0, 0], [2, 0, 0]],
                                   da_dt=sympy.Matrix([[0, 0, 0, 0, 0, 0, 0]])
        )

        correct_answer = sympy.Matrix([
            map(sympy.sympify, [
                '-0.06*y_0**2 + 0.03*y_0 + 2.0*y_0/(1 + y_2**2/c_0**2) + 1.0/(1 + y_2**2/c_0**2)',
                '-2.0*y_0/(c_0**2*(1 + y_2**2/c_0**2)**2) - 1.0/(c_0**2*(1 + y_2**2/c_0**2)**2) + 8.0*y_0*y_2**2/(c_0**4*(1 + y_2**2/c_0**2)**3) + 4.0*y_2**2/(c_0**4*(1 + y_2**2/c_0**2)**3)',
                0,
                0,
                '-4.0*y_2/(c_0**2*(1 + y_2**2/c_0**2)**2)',
                0,
                -0.06
               ])])

        self.assertEqual(answer, correct_answer)

    def test_da_dt_nonzero_case(self):
        """
        For an (actual) usage case of this program picked at random, where da_dt is non zero.
        The outputs should match up
        :return:
        """

        da_dt = sympy.Matrix([map(sympy.sympify,
                                  ['-0.03*y_0 + 1.0/(1 + y_2**2/c_0**2)',
                                   '-1.0/(c_0**2*(1 + y_2**2/c_0**2)**2) + 4.0*y_2**2/(c_0**4*(1 + y_2**2/c_0**2)**3)',
                                   0,
                                   0,
                                   0,
                                   0,
                                   0])])


        answer = self.fetch_answer(kvec=[1, 0, 0],
                                   ekcounter=[[1, 0, 0]],
                                   da_dt=da_dt)


        # Its weird that this is true so often:
        correct_answer = da_dt

        self.assertEqual(answer, correct_answer)