import unittest

import sympy

from means.model.model import Model
from means.approximation.lna.lna import LinearNoiseApproximation


class TestLNA(unittest.TestCase):

    def test_p53_model(self):
        stoichiometry_matrix = sympy.Matrix([[1, -1, -1, 0,  0,  0],
                                             [0,  0,  0, 1, -1,  0],
                                             [0,  0,  0, 0,  1, -1]])

        propensities = sympy.Matrix(['c_0',
                                     'c_1*y_0',
                                     'c_2*y_0*y_2/(c_6 + y_0)',
                                     'c_3*y_0',
                                     'c_4*y_1',
                                     'c_5*y_2'])

        species = sympy.Matrix(['y_0', 'y_1', 'y_2'])


        correct_rhs =  sympy.Matrix(
            ["c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)",
            "c_3*y_0 - c_4*y_1",
            "c_4*y_1 - c_5*y_2",
            "2*V_00*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_02*c_2*y_0/(c_6 + y_0) - V_20*c_2*y_0/(c_6 + y_0) + c_0**1.0 + (c_1*y_0)**1.0 + (c_2*y_0*y_2/(c_6 + y_0))**1.0",
            "V_00*c_3 - V_01*c_4 + V_01*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_21*c_2*y_0/(c_6 + y_0)",
            "V_01*c_4 - V_02*c_5 + V_02*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_22*c_2*y_0/(c_6 + y_0)",
            "V_00*c_3 - V_10*c_4 + V_10*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_12*c_2*y_0/(c_6 + y_0)",
            "V_01*c_3 + V_10*c_3 - 2*V_11*c_4 + (c_3*y_0)**1.0 + (c_4*y_1)**1.0",
            "V_02*c_3 + V_11*c_4 - V_12*c_4 - V_12*c_5 - (c_4*y_1)**1.0",
            "V_10*c_4 - V_20*c_5 + V_20*(-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0)) - V_22*c_2*y_0/(c_6 + y_0)",
            "V_11*c_4 + V_20*c_3 - V_21*c_4 - V_21*c_5 - (c_4*y_1)**1.0",
            "V_12*c_4 + V_21*c_4 - 2*V_22*c_5 + (c_4*y_1)**1.0 + (c_5*y_2)**1.0"])

        correct_lhs = sympy.Matrix(['y_0','y_1','y_2','V_00', 'V_01', 'V_02', 'V_10', 'V_11', 'V_12', 'V_20', 'V_21', 'V_22'])

        # todo use stub class?
        constants = ["c_{0}".format(i) for i in range(6)]
        model = Model(constants, species, propensities, stoichiometry_matrix)
        lna = LinearNoiseApproximation(model)
        problem = lna.run()

        answer_rhs = problem.right_hand_side
        answer_lhs = problem.left_hand_side



        self.assertEqual(correct_rhs, answer_rhs)
        self.assertEqual(correct_lhs, answer_lhs)
