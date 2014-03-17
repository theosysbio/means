import unittest

import sympy

from means.core.model import Model
from means.approximation.lna.lna import LinearNoiseApproximation
from means.util.sympyhelpers import to_sympy_matrix


class TestLNA(unittest.TestCase):

    def test_p53_model(self):
        stoichiometry_matrix = sympy.Matrix([[1, -1, -1, 0,  0,  0],
                                             [0,  0,  0, 1, -1,  0],
                                             [0,  0,  0, 0,  1, -1]])

        propensities = to_sympy_matrix(['c_0',
                                     'c_1*y_0',
                                     'c_2*y_0*y_2/(c_6 + y_0)',
                                     'c_3*y_0',
                                     'c_4*y_1',
                                     'c_5*y_2'])

        species = to_sympy_matrix(['y_0', 'y_1', 'y_2'])


        correct_rhs = to_sympy_matrix(
            ["c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)",
            "c_3*y_0 - c_4*y_1",
            "c_4*y_1 - c_5*y_2",
            "2*V_0_0*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_0_2*c_2*y_0/(c_6 + y_0) - V_2_0*c_2*y_0/(c_6 + y_0) + c_0**1.0 + (c_1*y_0)**1.0 + (c_2*y_0*y_2/(c_6 + y_0))**1.0",
            "V_0_0*c_3 - V_0_1*c_4 + V_0_1*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_2_1*c_2*y_0/(c_6 + y_0)",
            "V_0_1*c_4 - V_0_2*c_5 + V_0_2*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_2_2*c_2*y_0/(c_6 + y_0)",
            "V_0_0*c_3 - V_1_0*c_4 + V_1_0*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_1_2*c_2*y_0/(c_6 + y_0)",
            "V_0_1*c_3 + V_1_0*c_3 - 2*V_1_1*c_4 + (c_3*y_0)**1.0 + (c_4*y_1)**1.0",
            "V_0_2*c_3 + V_1_1*c_4 - V_1_2*c_4 - V_1_2*c_5 - (c_4*y_1)**1.0",
            "V_1_0*c_4 - V_2_0*c_5 + V_2_0*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_2_2*c_2*y_0/(c_6 + y_0)",
            "V_1_1*c_4 + V_2_0*c_3 - V_2_1*c_4 - V_2_1*c_5 - (c_4*y_1)**1.0",
            "V_1_2*c_4 + V_2_1*c_4 - 2*V_2_2*c_5 + (c_4*y_1)**1.0 + (c_5*y_2)**1.0"])

        correct_lhs = to_sympy_matrix(['y_0','y_1','y_2','V_0_0', 'V_0_1', 'V_0_2', 'V_1_0', 'V_1_1', 'V_1_2', 'V_2_0', 'V_2_1', 'V_2_2'])

        constants = ["c_{0}".format(i) for i in range(7)]
        model = Model(species, constants, propensities, stoichiometry_matrix)
        lna = LinearNoiseApproximation(model)
        problem = lna.run()

        answer_rhs = problem.right_hand_side
        answer_lhs = problem.left_hand_side



        self.assertEqual(correct_rhs, answer_rhs)
        self.assertEqual(correct_lhs, answer_lhs)
