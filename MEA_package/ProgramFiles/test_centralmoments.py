import unittest
import sympy
from centralmoments import eq_centralmoments

class CentralMomentsTestCase(unittest.TestCase):
    def test_centralmoments_using_p53model(self):
        counter = [[0, 0, 0], [0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]]

        mcounter = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 2], [0, 1, 1], [0, 2, 0],
                    [1, 0, 1], [1, 1, 0], [2, 0, 0]]

        m = sympy.Matrix([map(sympy.sympify,
                              ['c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)',
                               0,
                               0,
                               0,
                               'c_2*y_0/(c_6 + y_0)**2 - c_2/(c_6 + y_0)',
                               0,
                               '-c_2*y_0*y_2/(c_6 + y_0)**3 + c_2*y_2/(c_6 + y_0)**2']),
                          map(sympy.sympify, [
                              'c_3*y_0 - c_4*y_1',
                              0,
                              0,
                              0,
                              0,
                              0,
                              0]),
                          map(sympy.sympify, [
                              'c_4*y_1 - c_5*y_2',
                              0,
                              0,
                              0,
                              0,
                              0,
                              0
                          ])])

        taylor_m = None  # I don't think this is used

        number_of_variables = 3
        species = sympy.Matrix(map(sympy.var, ['y_0', 'y_1', 'y_2']))
        number_of_reactions = 6
        number_of_moments = 2
        propensities = sympy.Matrix(map(sympy.sympify, ['c_0',
                                                        'c_1 * y_0',
                                                        'c_2*y_0*y_2/(c_6 + y_0)',
                                                        'c_3*y_0',
                                                        'c_4*y_1',
                                                        'c_5*y_2']))

        stoichiometry_matrix = sympy.Matrix([[1, -1, -1, 0, 0, 0],
                                             [0, 0, 0, 1, -1, 0],
                                             [0, 0, 0, 0, 1, -1]])

        number_of_derivatives = number_of_moments

        answer = eq_centralmoments(counter, mcounter, m, taylor_m,
                                   number_of_variables, species, number_of_reactions,
                                   number_of_moments, propensities, stoichiometry_matrix, number_of_derivatives)


        # Note that this is a list, rather than a matrix
        correct_answer = [
            map(sympy.sympify, ['2*c_4*y_1*y_2 + c_4*y_1 - 2*c_5*y_2**2 + c_5*y_2 - 2*x001*(c_4*y_1 - c_5*y_2)',
                                '-2*c_5',
                                '2*c_4',
                                0,
                                0,
                                0,
                                0]),
            map(sympy.sympify, [
                'c_3*y_0*y_2 + c_4*y_1**2 - c_4*y_1*y_2 - c_4*y_1 - c_5*y_1*y_2 - x001*(c_3*y_0 - c_4*y_1) - x010*(c_4*y_1 - c_5*y_2)',
                0,
                '-c_4 - c_5',
                'c_4',
                'c_3',
                0,
                0]),
            map(sympy.sympify, ['2*c_3*y_0*y_1 + c_3*y_0 - 2*c_4*y_1**2 + c_4*y_1 - 2*x010*(c_3*y_0 - c_4*y_1)',
                                0,
                                0,
                                '-2*c_4',
                                0,
                                '2*c_3',
                                0]),
            map(sympy.sympify, [
                'c_0*y_2 - c_1*y_0*y_2 - c_2*y_0*y_2**2/(c_6 + y_0) + c_4*y_0*y_1 - c_5*y_0*y_2 - x001*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) - x100*(c_4*y_1 - c_5*y_2)',
                '-c_2*y_0/(c_6 + y_0)',
                0,
                0,
                '-c_1 + 2*c_2*y_0*y_2/(c_6 + y_0)**2 - 2*c_2*y_2/(c_6 + y_0) - c_5 - x001*(c_2*y_0/(c_6 + y_0)**2 - c_2/(c_6 + y_0))',
                'c_4',
                '-c_2*y_0*y_2**2/(c_6 + y_0)**3 + c_2*y_2**2/(c_6 + y_0)**2 - x001*(-c_2*y_0*y_2/(c_6 + y_0)**3 + c_2*y_2/(c_6 + y_0)**2)']),
            map(sympy.sympify, [
                'c_0*y_1 - c_1*y_0*y_1 - c_2*y_0*y_1*y_2/(c_6 + y_0) + c_3*y_0**2 - c_4*y_0*y_1 - x010*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)) - x100*(c_3*y_0 - c_4*y_1)',
                0,
                '-c_2*y_0/(c_6 + y_0)',
                0,
                'c_2*y_0*y_1/(c_6 + y_0)**2 - c_2*y_1/(c_6 + y_0) - x010*(c_2*y_0/(c_6 + y_0)**2 - c_2/(c_6 + y_0))',
                '-c_1 + c_2*y_0*y_2/(c_6 + y_0)**2 - c_2*y_2/(c_6 + y_0) - c_4',
                '-c_2*y_0*y_1*y_2/(c_6 + y_0)**3 + c_2*y_1*y_2/(c_6 + y_0)**2 + c_3 - x010*(-c_2*y_0*y_2/(c_6 + y_0)**3 + c_2*y_2/(c_6 + y_0)**2)'
            ]),
            map(sympy.sympify, [
                '2*c_0*y_0 + c_0 - 2*c_1*y_0**2 + c_1*y_0 - 2*c_2*y_0**2*y_2/(c_6 + y_0) + c_2*y_0*y_2/(c_6 + y_0) - 2*x100*(c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0))',
                0,
                0,
                0,
                '2*c_2*y_0**2/(c_6 + y_0)**2 - 4*c_2*y_0/(c_6 + y_0) - c_2*y_0/(c_6 + y_0)**2 + c_2/(c_6 + y_0) - 2*x100*(c_2*y_0/(c_6 + y_0)**2 - c_2/(c_6 + y_0))',
                0,
                '-2*c_1 - 2*c_2*y_0**2*y_2/(c_6 + y_0)**3 + 4*c_2*y_0*y_2/(c_6 + y_0)**2 + c_2*y_0*y_2/(c_6 + y_0)**3 - 2*c_2*y_2/(c_6 + y_0) - c_2*y_2/(c_6 + y_0)**2 - 2*x100*(-c_2*y_0*y_2/(c_6 + y_0)**3 + c_2*y_2/(c_6 + y_0)**2)'
            ])]

        self.assertEqual(answer, correct_answer)

    def test_centralmoments_using_MM_model(self):
        counter = [[0, 0], [0, 2], [1, 1], [2, 0]]

        mcounter = [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]]

        m = sympy.Matrix([map(sympy.sympify,
                              ['-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301)',
                               0,
                               '-c_0',
                               '-c_0']),
                          map(sympy.sympify, [
                              'c_2*(-y_0 - y_1 + 301)',
                              0,
                              0,
                              0]),
        ])

        taylor_m = None  # I don't think this is used

        number_of_variables = 2
        species = sympy.Matrix(map(sympy.var, ['y_0', 'y_1']))
        number_of_reactions = 3
        number_of_moments = 2
        propensities = sympy.Matrix(map(sympy.sympify, ['c_0*y_0*(y_0 + y_1 - 181)',
                                                        'c_1*(-y_0 - y_1 + 301)',
                                                        'c_2*(-y_0 - y_1 + 301)']))

        stoichiometry_matrix = sympy.Matrix([[-1, 1, 0],
                                             [0, 0, 1]])

        number_of_derivatives = number_of_moments

        answer = eq_centralmoments(counter, mcounter, m, taylor_m,
                                   number_of_variables, species, number_of_reactions,
                                   number_of_moments, propensities, stoichiometry_matrix, number_of_derivatives)


        # Note that this is a list, rather than a matrix
        correct_answer = [
            map(sympy.sympify, [
                '-2*c_2*x01*(-y_0 - y_1 + 301) + 2*c_2*y_1*(-y_0 - y_1 + 301) + c_2*(-y_0 - y_1 + 301)',
                '-2*c_2',
                '-2*c_2',
                0]),
            map(sympy.sympify, [
                '-c_0*y_0*y_1*(y_0 + y_1 - 181) + c_1*y_1*(-y_0 - y_1 + 301) - c_2*x10*(-y_0 - y_1 + 301) + c_2*y_0*(-y_0 - y_1 + 301) - x01*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301))',
                '-c_0*y_0 - c_1',
                'c_0*x01 - c_0*y_0 - c_0*y_1 - c_0*(y_0 + y_1 - 181) - c_1 - c_2',
                'c_0*x01 - c_0*y_1 - c_2'
            ]),
            map(sympy.sympify, [
                '-2*c_0*y_0**2*(y_0 + y_1 - 181) + c_0*y_0*(y_0 + y_1 - 181) + 2*c_1*y_0*(-y_0 - y_1 + 301) + c_1*(-y_0 - y_1 + 301) - 2*x10*(-c_0*y_0*(y_0 + y_1 - 181) + c_1*(-y_0 - y_1 + 301))',
                0,
                '2*c_0*x10 - 4*c_0*y_0 + c_0 - 2*c_1',
                '2*c_0*x10 - 4*c_0*y_0 - 2*c_0*(y_0 + y_1 - 181) + c_0 - 2*c_1'
            ])
        ]

        self.assertEqual(answer, correct_answer)