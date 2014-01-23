import unittest
import sympy
from LNA import LNA


class TestLNA(unittest.TestCase):

    def test_MM_model(self):
        stoichiometry_matrix = sympy.Matrix([[-1, 1, 0], [0, 0, 1]])
        propensities = sympy.Matrix(map(sympy.sympify, [
                                        'c_0*y_0*(y_0 + y_1 - 181)',
                                        'c_1*(-y_0 - y_1 + 301)',
                                        'c_2*(-y_0 - y_1 + 301)']))

        species = sympy.Matrix(map(sympy.sympify, ['y_0', 'y_1']))


        correct_dp_dt = stoichiometry_matrix * propensities
        correct_dv_dt = sympy.Matrix([
            ['2*V_00*(-c_0*y_0 - c_0*(y_0 + y_1 - 181) - c_1) + V_01*(-c_0*y_0 - c_1) + V_10*(-c_0*y_0 - c_1) + (c_1*(-y_0 - y_1 + 301))**1.0 + (c_0*y_0*(y_0 + y_1 - 181))**1.0',
             '-V_00*c_2 - V_01*c_2 + V_01*(-c_0*y_0 - c_0*(y_0 + y_1 - 181) - c_1) + V_11*(-c_0*y_0 - c_1)'],
            ['-V_00*c_2 - V_10*c_2 + V_10*(-c_0*y_0 - c_0*(y_0 + y_1 - 181) - c_1) + V_11*(-c_0*y_0 - c_1)',
             '-V_01*c_2- V_10*c_2 - 2*V_11*c_2 + (c_2*(-y_0 - y_1 + 301))**1.0']
        ])

        correct_v_list = map(sympy.sympify, ['V_00', 'V_01', 'V_10', 'V_11'])

        answer_dp_dt, answer_dv_dt, answer_v, _ = LNA(stoichiometry_matrix, propensities, species)

        self.assertEqual(correct_dp_dt, answer_dp_dt)
        self.assertEqual(correct_dv_dt, answer_dv_dt)
        # The v's are used only as a list later on, therefore let's compare them as lists as well
        self.assertEqual(correct_v_list, list(answer_v))