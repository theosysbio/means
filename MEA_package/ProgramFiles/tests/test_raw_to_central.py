import unittest
import sympy
from raw_to_central import raw_to_central

class TestRawToCentral(unittest.TestCase):

    def test_a_single_calculation_of_MXn_is_correct_for_mixed_moment(self):
        """
        Given a single n vector - a single mixed moment to compute values for,
        the function should return a single correct equation for that moment's value

        :return:
        """

        # Since I do not want to deal with whole matrix here, I provide only one set of possible values
        # Note that the function removes the first zero vector as of current code base, thus the leading zero vector
        n_values = [[0, 0, 0], [1, 0, 1]]

        # This is the unchanged result of the `mcounter` returned by `fcount(2,3)`
        possible_k_values = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                             [0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]
        means_of_species = [sympy.var('y_0'), sympy.var('y_1'), sympy.var('y_2')]

        right_hand_sides, _left_hand_sides = raw_to_central( n_values, means_of_species, possible_k_values)

        self.assertEqual(len(right_hand_sides), 1, "Was expecting to get back only one equation")

        # TODO: these symbols should not be hardcoded here as they may change provided change in other parts of code
        # I am not sure how to avoid this now though, so they stay.
        beta_terms = [sympy.Symbol('x_0_0_0'), sympy.Symbol('x_0_0_1'), sympy.Symbol('x_1_0_0'), sympy.Symbol('x_1_0_1')]

        # This is what the code should return: x_0_0_0*y_0*y_2 - x_0_0_1*y_0 - x_1_0_0*y_2 + x_1_0_1
        correct_answer = beta_terms[0] * means_of_species[0] * means_of_species[2] \
                         - beta_terms[1] * means_of_species[0] \
                         - beta_terms[2] * means_of_species[2] \
                         + beta_terms[3]

        self.assertEqual(right_hand_sides[0], correct_answer)

    def test_a_single_calculation_of_MXn_is_correct_for_mixed_moment_of_order_two(self):

        # Since I do not want to deal with whole matrix here, I provide only one set of possible values
        # Note that the function removes the first zero vector as of current code base, thus the leading zero vector
        n_values = [[0, 0, 0], [2, 0, 0]]

        # This is the unchanged result of the `mcounter` returned by `fcount(2,3)`
        possible_k_values = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                             [0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]
        means_of_species = [sympy.var('y_0'), sympy.var('y_1'), sympy.var('y_2')]

        right_hand_sides, left_hand_sides = raw_to_central( n_values, means_of_species, possible_k_values)

        self.assertEqual(len(right_hand_sides), 1, "Was expecting to get back only one equation")

        # TODO: these symbols should not be hardcoded here as they may change provided change in other parts of code
        # I am not sure how to avoid this now though, so they stay.
        beta_terms = [sympy.Symbol('x_0_0_0'), sympy.Symbol('x_1_0_0'), sympy.Symbol('x_2_0_0')]

        # This is what the code should return: x_0_0_0*y_0**2 - 2*x_1_0_0*y_0 + x_2_0_0
        correct_answer = beta_terms[0] * (means_of_species[0]**2) \
                         - 2 * beta_terms[1] * means_of_species[0] \
                         + beta_terms[2]
        self.assertEqual(right_hand_sides[0], correct_answer)


    def test_whole_calculation_of_second_order_moments_of_two_species(self):
        """
        Given a run configuration of n and k variable posibilities as in model_NN.txt the output of the
        function should return the correct equations
        :return:
        """

        n_values = [[0, 0], [0, 2], [1, 1], [2, 0]]
        possible_k_values = n_values + [[1, 0], [0, 1]]
        means_of_species = [sympy.var('y_0'), sympy.var('y_1')]

        right_hand_sides, left_hand_sides = raw_to_central( n_values, means_of_species, possible_k_values)
        self.assertEqual(len(right_hand_sides), 3, "Was expecting to get back three equations, one for each n_value except for zero vector")

        beta_terms = {'00': sympy.Symbol('x_0_0'),
                      '01': sympy.Symbol('x_0_1'),
                      '10': sympy.Symbol('x_1_0'),
                      '11': sympy.Symbol('x_1_1'),
                      '02': sympy.Symbol('x_0_2'),
                      '20': sympy.Symbol('x_2_0')}

        correct_answers = [
            beta_terms['00'] * means_of_species[1]**2 - 2 * beta_terms['01'] * means_of_species[1] + beta_terms['02'],
            beta_terms['00'] * means_of_species[0] * means_of_species[1] \
                - beta_terms['01'] * means_of_species[0] - beta_terms['10'] * means_of_species[1] + beta_terms['11'],
            beta_terms['00'] * means_of_species[0]**2 - 2 * beta_terms['10'] * means_of_species[0] + beta_terms['20']
        ]

        for correct_answer, actual_answer in zip(correct_answers, right_hand_sides):
            self.assertEqual(correct_answer, actual_answer)
