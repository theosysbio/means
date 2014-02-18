import unittest
from numpy.testing import assert_array_almost_equal
from sympy import Symbol, Float, MutableDenseMatrix
from means.approximation.ode_problem import Moment, ODEProblem
import numpy as np
from means.inference import ParameterInference
from means.inference.sumsq_infer import i0_to_test
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
        self.observed_timepoints = np.arange(0, 20, 0.5)

    def test_sum_of_squares(self):

        params_with_variability = [(0.001, True), (0.5, True), (330.0, True)]
        initcond_with_variability = [(320.0, True), (0, False)]

        optimiser_method = 'sum_of_squares'
        constraints = None

        inference = ParameterInference(self.dimer_problem,
                                       params_with_variability,
                                       initcond_with_variability,
                                       constraints,
                                       self.observed_timepoints,
                                       self.observed_trajectories,
                                       method=optimiser_method)
        parameters, distance, iterations, evaluations, __ = inference.infer()

        (opt_param, opt_initconds) = i0_to_test(list(parameters), params_with_variability, initcond_with_variability)

        assert_array_almost_equal(opt_param, [0.00012707216279026558, 0.089221047933167152, 301.09712661982326], decimal=2)
        assert_array_almost_equal(opt_initconds, [301.00385505534678, 0], decimal=1)
        self.assertAlmostEqual(distance, 0.00823811223489, delta=5e-3)

    def test_sum_of_squares_means_only(self):

        params_with_variability = [(0.001, True), (0.5, True), (330.0, True)]
        initcond_with_variability = [(320.0, True), (0, False)]

        optimiser_method = 'sum_of_squares'
        constraints = None

        inference = ParameterInference(self.dimer_problem,
                                       params_with_variability,
                                       initcond_with_variability,
                                       constraints,
                                       self.observed_timepoints,
                                       # Only means trajectory
                                       [self.observed_trajectories[0]],
                                       method=optimiser_method)
        parameters, distance, iterations, evaluations, __ = inference.infer()

        (opt_param, opt_initconds) = i0_to_test(list(parameters), params_with_variability, initcond_with_variability)

        assert_array_almost_equal(opt_param, [0.00017664681741244679, 0.043856181172598596, 495.49530645744187], decimal=2)
        assert_array_almost_equal(opt_initconds, [301.27426184772685, 0], decimal=1)
        self.assertAlmostEqual(distance, 0.350944811744, delta=5e-3)



