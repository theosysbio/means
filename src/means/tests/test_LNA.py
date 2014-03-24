import unittest

import sympy

from means.core.model import Model
from means.approximation.lna.lna import LinearNoiseApproximation
from means.util.sympyhelpers import to_sympy_matrix, assert_sympy_expressions_equal


class TestLNA(unittest.TestCase):

    def test_transcription_model(self):
        #use simple production and degradation of mRNA and protein for testing
        # mRNA production rate is k1, degradation rate is g1
        # protein production rate is k2, degradation rate is g2
        stoichiometry_matrix = sympy.Matrix([[1, -1, 0,  0],
                                             [0,  0, 1, -1]])

        propensities = to_sympy_matrix(['k1',
                                     'g1*x',
                                     'k2*x',
                                     'g2*y'])

        species = to_sympy_matrix(['x', 'y'])


        correct_rhs = to_sympy_matrix(
            ["k1 - g1 * x",
            "k2 * x - g2 * y",
            "k1 + g1 * x - 2 * g1 * V_0_0",
            "k2 * V_0_0 - (g1 + g2) * V_0_1",
            "k2 * x + g2 * y + k2 * V_0_1 + k2 * V_0_1 - 2 * g2 * V_1_1"])

        correct_lhs = to_sympy_matrix(['x','y','V_0_0', 'V_0_1', 'V_1_1'])

        constants = ["k1","k2","g1","g2"]
        model = Model(species, constants, propensities, stoichiometry_matrix)
        lna = LinearNoiseApproximation(model)
        problem = lna.run()

        answer_rhs = problem.right_hand_side
        answer_lhs = problem.left_hand_side

        assert_sympy_expressions_equal(correct_rhs, answer_rhs)
        self.assertEqual(correct_lhs, answer_lhs)

