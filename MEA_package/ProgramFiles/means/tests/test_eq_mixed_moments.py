import unittest

import sympy

from means.approximation.mea.eq_mixedmoments import make_f_of_x
from means.approximation.mea.eq_mixedmoments import make_f_expectation
from means.approximation.mea.eq_mixedmoments import make_k_chose_e
from means.approximation.mea.eq_mixedmoments import make_s_pow_e
from means.approximation.mea.eq_mixedmoments import eq_mixedmoments


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
        KWARGS = {
                  'amat': self.PROPENSITIES,
                  'counter': self.COUNTER,
                  'S': self.STOICHIOMETRY_MATRIX,
                  'ymat': self.SPECIES,
                  'k_vec': kvec,
                  'ek_counter': ekcounter,
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

class TestUtilFunctions(AbstractTestEqMixedMoments):

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


    def test_make_f_expectation(self):

        """
        Given the vectors of variable names,
        Given the vector of combination of moment orders "counter"
        Given the specified reaction equation,
        Then resulting vector Should be exactly as expected

        :return:
        """

        variables = sympy.Matrix(["y_0", "y_1", "y_2"])
        expr = sympy.S("(y_0 + c_0 * y_1 )/y_2")
        counter = self.COUNTER

        result = make_f_expectation(variables, expr, counter)
        expected_result = sympy.Matrix(["(c_0*y_1 + y_0)/y_2", "(c_0*y_1 + y_0)/y_2**3", "-c_0/y_2**2", "0", "-1/y_2**2", "0", "0"])
        self.assertEqual(result, expected_result)

    def test_make_k_chose_e(self):
        """
        Given the vectors k and e ,
        Then result Should be exactly as specified

        :return:
        """

        test_a = {"e_vec":[1,2,0], "k_vec":[1,2,0]}
        expected_a = 1
        test_b = {"e_vec":[1,2,0], "k_vec":[3,3,3]}
        expected_b = 9

        result_a = make_k_chose_e(**test_a)
        result_b = make_k_chose_e(**test_b)

        self.assertEqual(result_a, expected_a)
        self.assertEqual(result_b, expected_b)


    def test_make_s_pow_e(self):

        """
        Given the vector e and,
        Given the to stoichio. toy matrix,
        Then result Should be exactly as specified

        :return:
        """

        stoichiometry_matrix = sympy.Matrix([[-1], [3], [1]])
        e_vec = [3, 2, 2]
        expected_result = (-1 ** 3) * (3 ** 2) * (1 ** 2)
        result = make_s_pow_e(stoichiometry_matrix, 0, e_vec)
        self.assertEqual(result, expected_result)

    def test_make_f_of_x(self):

        """
        Given the vectors k and e and the specified reaction equation,
        Then result Should be exactly as specified

        :return:
        """

        e_vec = [1,2,0]
        k_vec = [3,3,3]
        variables = sympy.Matrix(["y_0", "y_1", "y_2"])
        reaction = sympy.S("(y_0 + c_0 * y_1 )/y_2")
        expected_result = sympy.S("y_0**2*y_1*y_2**2*(c_0*y_1 + y_0)")
        result =  make_f_of_x(variables, k_vec, e_vec, reaction)

        self.assertEqual(result, expected_result)
