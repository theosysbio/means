import unittest
from numpy.testing import assert_array_almost_equal
from sympy import Symbol, Float, MutableDenseMatrix
from means.approximation.ode_problem import Moment, ODEProblem
import numpy as np
from means.inference import Inference
# We need renaming as otherwise nose picks it up as a test
from means.simulation import Trajectory


class TestInferenceForRegressions(unittest.TestCase):

    def setUp(self):
        # Create an ODEProblem from dimer instance
        c_0 = Symbol('c_0')
        c_1 = Symbol('c_1')
        c_2 = Symbol('c_2')

        self.constants = [c_0, c_1, c_2]

        y_0 = Symbol('y_0')

        yx1 = Symbol('yx1')

        right_hand_side = MutableDenseMatrix([[-2*c_0*y_0*(y_0 - 1) - 2*c_0*yx1 + 2*c_1*(Float('0.5', prec=15)*c_2 - Float('0.5', prec=15)*y_0)], [Float('4.0', prec=15)*c_0*y_0**2 - Float('4.0', prec=15)*c_0*y_0 + Float('2.0', prec=15)*c_1*c_2 - Float('2.0', prec=15)*c_1*y_0 - yx1*(Float('8.0', prec=15)*c_0*y_0 - Float('8.0', prec=15)*c_0 + Float('2.0', prec=15)*c_1)]])

        ode_lhs_terms = [Moment(np.array([1]), symbol=y_0), Moment(np.array([2]), symbol=yx1)]

        self.dimer_problem = ODEProblem('MEA', ode_lhs_terms, right_hand_side, self.constants)
        timepoints = np.arange(0, 20, 0.5)
        self.observed_trajectories = [Trajectory(timepoints,
                                                 [301., 290.1919552, 280.58701279, 272.03059275, 264.39179184
                                                     , 257.55906225, 251.43680665, 245.94267027, 241.0053685,
                                                  236.56293031
                                                     , 232.56126687, 228.95299703, 225.69647718, 222.75499485,
                                                  220.09609439
                                                     , 217.69101017, 215.51418738, 213.54287512, 211.75677913,
                                                  210.13776407
                                                     , 208.6695974, 207.33772795, 206.12909392, 205.03195581,
                                                  204.03575051
                                                     , 203.13096347, 202.30901645, 201.5621687, 200.88342959,
                                                  200.26648137
                                                     , 199.70561061, 199.19564728, 198.73191049, 198.3101601,
                                                  197.92655342
                                                     , 197.57760651, 197.26015951, 196.97134545, 196.70856234,
                                                  196.46944793],
                                                 Moment([1])),
                                      Trajectory(timepoints,
                                                 [
                                                     0., 20.10320788, 35.54689328, 47.51901615, 56.88242563
                                                     , 64.26983231, 70.14921364, 74.86943532, 78.69244584, 81.81623963
                                                     , 84.39139622, 86.53309942, 88.32994353, 89.85043353, 91.1478162
                                                     , 92.26369294, 93.23073689, 94.07474712, 94.81620873, 95.47148295
                                                     , 96.05371852, 96.57355199, 97.03964777, 97.45911583, 97.83783557
                                                     , 98.18070779, 98.49185119, 98.77475594, 99.03240415, 99.26736458
                                                     , 99.48186728, 99.67786273, 99.85706877, 100.02100803, 100.17103799
                                                     , 100.3083751, 100.43411443, 100.54924568, 100.65466632,
                                                     100.75119251],
                                                 Moment([2]))]

    def test_params_with_variability_creation(self):

        class InferenceStub(Inference):
            def __init__(self):
                pass


        pi_stub = InferenceStub()

        def compare(variables, correct_values_with_variability, correct_constraints):

            values_with_variability, constraints = pi_stub._generate_values_with_variability_and_constraints(
                self.dimer_problem.constants,
                parameters,
                variables)

            self.assertEqual(values_with_variability, correct_values_with_variability)
            self.assertEqual(constraints, correct_constraints)

        parameters = [0.001, 0.5, 330.0]

        expected_values_with_variability = zip(parameters, [False, True, True])
        expected_constraints = [None, (329, 330)]

        symbol_keys = {Symbol('c_1'): None, Symbol('c_2'): (329, 330)}
        compare(symbol_keys, expected_values_with_variability, expected_constraints)

    def test_initialisation_of_variable_parameters(self):
        parameters = [0.001, 0.5, 330.0]
        initial_conditions = [320.0, 0]

        def check_initialisation(variable_parameters, expected_parameters_with_variability,
                                   expected_initial_conditions_with_variability, expected_constraints):
            p = Inference(self.dimer_problem, parameters, initial_conditions,
                                   variable_parameters, [Trajectory([1, 2, 3], [1,2,3], 'x')])

            self.assertEquals(p.starting_parameters_with_variability, expected_parameters_with_variability)
            self.assertEqual(p.starting_conditions_with_variability, expected_initial_conditions_with_variability)
            self.assertEqual(p.constraints, expected_constraints)

        symbol_keys = {Symbol('c_1'): None, Symbol('c_2'): (329, 330), Symbol('y_0'): [319, 321]}
        check_initialisation(symbol_keys, zip(parameters, [False, True, True]), zip(initial_conditions, [True, False]),
                             [None, (329, 330), (319, 321)])

        string_keys = {'c_1': None, 'c_2': (329, 330), 'y_0': [319, 321]}
        check_initialisation(string_keys, zip(parameters, [False, True, True]), zip(initial_conditions, [True, False]),
                             [None, (329, 330), (319, 321)])

        mixed_keys = {'c_1': None, Symbol('c_2'): (329, 330), Symbol('y_0'): [319, 321]}
        check_initialisation(mixed_keys, zip(parameters, [False, True, True]), zip(initial_conditions, [True, False]),
                             [None, (329, 330), (319, 321)])

        set_keys = {'c_0', 'c_1'}
        check_initialisation(set_keys, zip(parameters, [True, True, False]), zip(initial_conditions, [False, False]),
                             [None, None])

        list_keys = ['c_2']
        check_initialisation(list_keys, zip(parameters, [False, False, True]), zip(initial_conditions, [False, False]),
                             [None])

    def test_initialisation_with_no_variable_parameters_specified_fails(self):
        """
        Given that we are trying to initialise our Inference with no variable parameters
        We should raise a ValueError as, well, there is no variables to work with then, is there?
        """
        self.assertRaises(ValueError, Inference, self.dimer_problem, [0.001, 0.5, 330.0], [320.0, 0],
                          None,  # No variable parameters
                          [1, 2, 3], [1, 2, 3])

        self.assertRaises(ValueError, Inference, self.dimer_problem, [0.001, 0.5, 330.0], [320.0, 0],
                          [],  # No variable parameters
                          [1, 2, 3], [1, 2, 3])

        self.assertRaises(ValueError, Inference, self.dimer_problem, [0.001, 0.5, 330.0], [320.0, 0],
                          {},  # No variable parameters
                          [1, 2, 3], [1, 2, 3])

        self.assertRaises(ValueError, Inference, self.dimer_problem, [0.001, 0.5, 330.0], [320.0, 0],
                          set(),  # No variable parameters
                          [1, 2, 3], [1, 2, 3])

    def test_initialisation_with_parameters_that_do_not_exist_fails(self):
        """
        Given some variable parameters that do not exist in ODE, the intialisation of Inference should fail
        with KeyError
        """

        self.assertRaises(KeyError, Inference, self.dimer_problem, [0.001, 0.5, 330.0], [320.0, 0],
                          {'y_0': None, 'y_1': (1, 2)},  # y_1 is nowhere to be found in the Problem parameters
                          [1, 2, 3], [1, 2, 3])

        self.assertRaises(KeyError, Inference, self.dimer_problem, [0.001, 0.5, 330.0], [320.0, 0],
                          ['y_0', 'y_1'],  # y_1 is nowhere to be found in the Problem parameters
                          [1, 2, 3], [1, 2, 3])

        self.assertRaises(KeyError, Inference, self.dimer_problem, [0.001, 0.5, 330.0], [320.0, 0],
                          {'y_0': None, Symbol('y_1'): (1, 2)},  # y_1 is nowhere to be found in the Problem parameters
                          [1, 2, 3], [1, 2, 3])
        self.assertRaises(KeyError, Inference, self.dimer_problem, [0.001, 0.5, 330.0], [320.0, 0],
                          ['y_0', Symbol('y_1')],  # y_1 is nowhere to be found in the Problem parameters
                          [1, 2, 3], [1, 2, 3])

    def test_initialisation_fails_with_funny_ranges(self):
        """
        Given initialisation of Inference with a variable whose range is not None or a two-element iterable,
        initialisation should fail with ValueError
        """

        self.assertRaises(ValueError, Inference, self.dimer_problem, [0.001, 0.5, 330.0], [320.0, 0],
                          {'y_0': None, 'c_0': (1, 2, 3)},  # 1, 2, 3 is not a range
                          [1, 2, 3], [1, 2, 3])

        self.assertRaises(ValueError, Inference, self.dimer_problem, [0.001, 0.5, 330.0], [320.0, 0],
                          {'y_0': None, 'c_0': [1]},  # neither is 1
                          [1, 2, 3], [1, 2, 3])


        self.assertRaises(ValueError, Inference, self.dimer_problem, [0.001, 0.5, 330.0], [320.0, 0],
                          {'y_0': None, 'c_0': 'from one to two'},  # no string parsing just now
                          [1, 2, 3], [1, 2, 3])

        self.assertRaises(ValueError, Inference, self.dimer_problem, [0.001, 0.5, 330.0], [320.0, 0],
                          {'y_0': None, 'c_0': 'xy'},  # Two letter strings should also fail
                          [1, 2, 3], [1, 2, 3])

    def test_sum_of_squares(self):

        parameters = [0.001, 0.5, 330.0]
        initial_conditions = [320.0, 0]

        optimiser_method = 'sum_of_squares'
        variable_parameters = ['c_0', 'c_1', 'c_2', 'y_0']

        inference = Inference(self.dimer_problem,
                                       parameters,
                                       initial_conditions,
                                       variable_parameters,
                                       self.observed_trajectories,
                                       method=optimiser_method)
        inference_result = inference.infer()

        assert_array_almost_equal(inference_result.optimal_parameters, [0.00012707365867374723, 0.089230125524899603, 301.09267270531382])
        assert_array_almost_equal(inference_result.optimal_initial_conditions, [300.986186470956, 0])
        self.assertAlmostEqual(inference_result.distance_at_minimum, 0.0107977081308)

    def test_sum_of_squares_means_only(self):

        parameters = [0.001, 0.5, 330.0]
        initial_conditions = [320.0, 0]

        optimiser_method = 'sum_of_squares'
        variable_parameters = ['c_0', 'c_1', 'c_2', 'y_0']

        inference = Inference(self.dimer_problem,
                                       parameters,
                                       initial_conditions,
                                       variable_parameters,
                                       # Only means trajectory
                                       [self.observed_trajectories[0]],
                                       method=optimiser_method)
        inference_result = inference.infer()

        assert_array_almost_equal(inference_result.optimal_parameters, [0.00017664682228204413, 0.043856182869604673,
                                                                        495.49530551533815])
        assert_array_almost_equal(inference_result.optimal_initial_conditions, [301.27426546880224, 0])
        self.assertAlmostEqual(inference_result.distance_at_minimum, 0.350924941344)

    def test_gamma_inference(self):
        starting_params = [0.0003553578523702354, 0.29734640303161364, 306.2260484701648]
        starting_initial_conditions = [304.7826314512718, 0.0]

        variable_parameters = {'c_0': (0.0, 0.001),
                               'c_1': (0.0, 0.5),
                               'c_2': (260.0, 330.0),
                               'y_0': (290.0, 320.0)}

        optimiser_method = 'gamma'

        inference = Inference(self.dimer_problem,
                                       starting_params,
                                       starting_initial_conditions,
                                       variable_parameters,
                                       # Only means trajectory
                                       [self.observed_trajectories[0]],
                                       method=optimiser_method)

        inference_result = inference.infer()

        assert_array_almost_equal(inference_result.optimal_parameters, [9.8148438195906734e-05, 0.11551859499768752,
                                                                        260.00000014956925])
        assert_array_almost_equal(inference_result.optimal_initial_conditions, [300.51956949425931, 0])
        self.assertAlmostEqual(inference_result.distance_at_minimum, 115.362403987, places=3)

    def test_normal_inference(self):
        starting_params = [0.0003553578523702354, 0.29734640303161364, 306.2260484701648]
        starting_conditions = [304.7826314512718, 0]

        optimiser_method = 'normal'
        variable_parameters = {'c_0': (0.0, 0.001),
                               'c_1': (0.0, 0.5),
                               'c_2': (260.0, 330.0),
                               'y_0': (290.0, 320.0)}

        inference = Inference(self.dimer_problem,
                                       starting_params,
                                       starting_conditions,
                                       variable_parameters,
                                       # Only means trajectory
                                       [self.observed_trajectories[0]],
                                       method=optimiser_method)

        inference_result = inference.infer()

        assert_array_almost_equal(inference_result.optimal_parameters, [9.5190703395740974e-05, 0.10494581837857614,
                                                                        260.00255131904339])
        assert_array_almost_equal(inference_result.optimal_initial_conditions, [298.81392432779984, 0])
        self.assertAlmostEqual(inference_result.distance_at_minimum, 115.687969964)

    def test_lognormal_inference(self):
        starting_params = [0.0008721146403084233, 0.34946447118966373, 285.8232870026351]
        starting_conditions = [309.6216092798371, 0]

        optimiser_method = 'lognormal'
        variable_parameters = {'c_0': (0.0, 0.001),
                               'c_1': (0.0, 0.5),
                               'c_2': (260.0, 330.0),
                               'y_0': (290.0, 320.0)}

        inference = Inference(self.dimer_problem,
                                       starting_params,
                                       starting_conditions,
                                       variable_parameters,
                                       # Only means trajectory
                                       [self.observed_trajectories[0]],
                                       method=optimiser_method)

        inference_result = inference.infer()

        assert_array_almost_equal(inference_result.optimal_parameters, [0.00097039430700166115, 9.1893721957377865e-07,
                                                                        303.48309650132126])
        assert_array_almost_equal(inference_result.optimal_initial_conditions, [290.06297620238149, 0])
        self.assertAlmostEqual(inference_result.distance_at_minimum, 2090.53271923)



