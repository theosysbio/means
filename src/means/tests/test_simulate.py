import unittest
import means
from means.util.sympyhelpers import to_sympy_matrix
from means.approximation.ode_problem import ODEProblem, ODETermBase, Moment, VarianceTerm
from means.simulation import Simulation
from numpy.testing import assert_array_almost_equal
import numpy as np
from sympy import Symbol, MutableDenseMatrix, symbols

class ConstantDerivativesProblem(ODEProblem):
    def __init__(self):
        super(ConstantDerivativesProblem, self).__init__(method=None,
                                                         ode_lhs_terms=[ODETermBase('y_1'), ODETermBase('y_2')],
                                                         right_hand_side=['c_1', 'c_2'],
                                                         constants=['c_1', 'c_2'])

class TestSimulate(unittest.TestCase):

    def test_simulation_of_simple_model(self):
        """
        Given the simplest possible problem, the one with constant derivatives,
        results produced by the simulation should be easily predictable.
        """
        s = Simulation(ConstantDerivativesProblem())

        trajectories = s.simulate_system(parameters=[0, 1],
                                         initial_conditions=[3, 2],
                                         timepoints=[0, 1, 2, 3])
        trajectories_dict = {trajectory.description.symbol: trajectory for trajectory in trajectories}
        y_1_trajectory = trajectories_dict['y_1']
        y_2_trajectory = trajectories_dict['y_2']

        assert_array_almost_equal(y_1_trajectory.values, [3, 3, 3, 3])
        assert_array_almost_equal(y_2_trajectory.values, [2, 3, 4, 5])

    def test_postprocessing_for_lna_model(self):
        """
        Given that the problem we are modelling is LNA, check that the results are sampled from a gaussian distribution.

        TODO: Write a test for more than two species, currently that is broken in LNA.
        """

        lna_for_lotka_volterra = ODEProblem(method='LNA',
                                            ode_lhs_terms=[Moment([1, 0], symbol='Pred'),
                                                           Moment([0, 1], symbol='Prey'),
                                                           VarianceTerm('V_00', (0, 0)),
                                                           VarianceTerm('V_01', (0, 1)),
                                                           VarianceTerm('V_10', (1, 0)),
                                                           VarianceTerm('V_11', (1, 1))],
                                            right_hand_side=to_sympy_matrix(['Pred*Prey*k_2 - Pred*k_3',
                                                                             '-Pred*Prey*k_2 + Prey*k_1',
                                                                             'Pred*V_01*k_2 + Pred*V_10*k_2 + 2*V_00*(Prey*k_2 - k_3) + (Pred*k_3)**1.0 + (Pred*Prey*k_2)**1.0',
                                                                             'Pred*V_11*k_2 - Prey*V_00*k_2 + V_01*(-Pred*k_2 + k_1) + V_01*(Prey*k_2 - k_3) - (Pred*Prey*k_2)**1.0',
                                                                             'Pred*V_11*k_2 - Prey*V_00*k_2 + V_10*(-Pred*k_2 + k_1) + V_10*(Prey*k_2 - k_3) - (Pred*Prey*k_2)**1.0',
                                                                             '-Prey*V_01*k_2 - Prey*V_10*k_2 + 2*V_11*(-Pred*k_2 + k_1) + (Prey*k_1)**1.0 + (Pred*Prey*k_2)**1.0']),
                                            constants=['k_1', 'k_2', 'k_3'])

        s = Simulation(lna_for_lotka_volterra)
        np.random.seed(42)

        trajectories = s.simulate_system(range(3), [200, 10], [1, 2, 3, 4, 5])

        trajectories_dict = {trajectory.description.symbol: trajectory for trajectory in trajectories}

        assert_array_almost_equal(trajectories_dict['Pred'].values, np.array([2.00000000e+02, 2.52230012e+01,
                                                                              4.30324933e+00, -6.17462164e-01,
                                                                              1.95071783e-01]))

        assert_array_almost_equal(trajectories_dict['Prey'].values, np.array([1.00000000e+01, -1.77761036e-02,
                                                                              3.65911320e-03, -2.78905874e-03,
                                                                              -3.02892609e-03]))

class TestSimulateRegressionForPopularModels(unittest.TestCase):

    def test_p53_3_moments(self):

        # This is just a hardcoded result of MomentExpansionApproximation(MODEL_P53,3).run()
        y_0, y_1, y_2 = symbols(['y_0', 'y_1', 'y_2'])

        yx1, yx2, yx3, yx4, yx5, yx6 = symbols(['yx1', 'yx2', 'yx3', 'yx4', 'yx5', 'yx6'])
        yx7, yx8, yx9, yx10, yx11, yx12 = symbols(['yx7', 'yx8', 'yx9', 'yx10', 'yx11', 'yx12'])
        yx13, yx14, yx15, yx16 = symbols(['yx13', 'yx14', 'yx15', 'yx16'])

        c_0, c_1, c_2, c_3, c_4, c_5, c_6 = symbols(['c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6'])

        ode_lhs_terms = [Moment(np.array([1, 0, 0]), symbol=y_0),
                         Moment(np.array([0, 1, 0]), symbol=y_1),
                         Moment(np.array([0, 0, 1]), symbol=y_2),
                         Moment(np.array([0, 0, 2]), symbol=yx1),
                         Moment(np.array([0, 0, 3]), symbol=yx2),
                         Moment(np.array([0, 1, 1]), symbol=yx3),
                         Moment(np.array([0, 1, 2]), symbol=yx4),
                         Moment(np.array([0, 2, 0]), symbol=yx5),
                         Moment(np.array([0, 2, 1]), symbol=yx6),
                         Moment(np.array([0, 3, 0]), symbol=yx7),
                         Moment(np.array([1, 0, 1]), symbol=yx8),
                         Moment(np.array([1, 0, 2]), symbol=yx9),
                         Moment(np.array([1, 1, 0]), symbol=yx10),
                         Moment(np.array([1, 1, 1]), symbol=yx11),
                         Moment(np.array([1, 2, 0]), symbol=yx12),
                         Moment(np.array([2, 0, 0]), symbol=yx13),
                         Moment(np.array([2, 0, 1]), symbol=yx14),
                         Moment(np.array([2, 1, 0]), symbol=yx15),
                         Moment(np.array([3, 0, 0]), symbol=yx16)]

        constants = [c_0, c_1, c_2, c_3, c_4, c_5, c_6]

        right_hand_side = MutableDenseMatrix([[(-c_2*c_6*y_2*yx16 - c_2*c_6*yx8*(c_6 + y_0)**2 + c_2*c_6*(c_6 + y_0)*(y_2*yx13 + yx14) - c_2*y_0*y_2*(c_6 + y_0)**3 + (c_0 - c_1*y_0)*(c_6 + y_0)**4)/(c_6 + y_0)**4], [c_3*y_0 - c_4*y_1], [c_4*y_1 - c_5*y_2], [c_4*y_1 + 2*c_4*yx3 + c_5*y_2 - 2*c_5*yx1], [c_4*y_1 + 3*c_4*yx3 + 3*c_4*yx4 - c_5*y_2 + 3*c_5*yx1 - 3*c_5*yx2], [c_3*yx8 - c_4*y_1 + c_4*yx5 - yx3*(c_4 + c_5)], [c_3*yx9 - c_4*y_1 - 2*c_4*yx3 - c_4*yx4 + c_4*yx5 + 2*c_4*yx6 + c_5*yx3 - 2*c_5*yx4], [c_3*y_0 + 2*c_3*yx10 + c_4*y_1 - 2*c_4*yx5], [2*c_3*yx11 + c_3*yx8 + c_4*y_1 + c_4*yx3 - 2*c_4*yx5 - 2*c_4*yx6 + c_4*yx7 - c_5*yx6], [c_3*y_0 + 3*c_3*yx10 + 3*c_3*yx12 - c_4*y_1 + 3*c_4*yx5 - 3*c_4*yx7], [(c_2*c_6*y_2*yx14 - c_2*y_0*yx1*(c_6 + y_0)**2 + c_4*yx10*(c_6 + y_0)**3 - (c_6 + y_0)*(c_2*c_6*yx9 + yx8*(c_2*c_6*y_2 + (c_1 + c_5)*(c_6 + y_0)**2)))/(c_6 + y_0)**3], [(c_2*c_6*y_2*yx1*yx16 - c_2*c_6*yx1*(c_6 + y_0)*(y_2*yx13 + yx14) + c_4*(c_6 + y_0)**4*(yx10 + 2*yx11) + (c_6 + y_0)**3*(-c_0*c_6*yx1 - c_0*y_0*yx1 + c_1*c_6*y_0*yx1 + c_1*y_0**2*yx1 + c_2*y_0*y_2*yx1 - c_2*y_0*yx2 - 2*c_4*c_6*y_1*yx8 - 2*c_4*y_0*y_1*yx8 + 2*c_5*c_6*y_2*yx8 + 2*c_5*y_0*y_2*yx8 - yx1*(c_2*y_0*y_2 - (c_0 - c_1*y_0)*(c_6 + y_0))) + (c_6 + y_0)**2*(yx8*(c_2*c_6*yx1 + (c_6 + y_0)**2*(2*c_4*y_1 - 2*c_5*y_2 + c_5)) - yx9*(c_2*c_6*y_2 + (c_1 + 2*c_5)*(c_6 + y_0)**2)))/(c_6 + y_0)**4], [(c_2*c_6*y_2*yx15 - c_2*y_0*yx3*(c_6 + y_0)**2 + c_3*yx13*(c_6 + y_0)**3 - (c_6 + y_0)*(c_2*c_6*yx11 + yx10*(c_2*c_6*y_2 + (c_1 + c_4)*(c_6 + y_0)**2)))/(c_6 + y_0)**3], [(c_2*c_6*y_2*yx16*yx3 + (c_6 + y_0)**4*(c_4*yx12 - yx10*(-c_4*y_1 + c_4 + c_5*y_2)) + (c_6 + y_0)**3*(-c_0*c_6*yx3 - c_0*y_0*yx3 + c_1*c_6*y_0*yx3 + c_1*y_0**2*yx3 + c_2*y_0*y_2*yx3 - c_2*y_0*yx4 - c_3*c_6*y_0*yx8 - c_3*y_0**2*yx8 - c_4*c_6*y_1*yx10 + c_4*c_6*y_1*yx8 - c_4*y_0*y_1*yx10 + c_4*y_0*y_1*yx8 + c_5*c_6*y_2*yx10 + c_5*y_0*y_2*yx10 - yx3*(-c_0*c_6 - c_0*y_0 + c_1*c_6*y_0 + c_1*y_0**2 + c_2*y_0*y_2)) + (c_6 + y_0)**2*(-yx11*(c_2*c_6*y_2 + (c_6 + y_0)**2*(c_1 + c_4 + c_5)) + yx8*(c_2*c_6*yx3 + (c_6 + y_0)**2*(c_3*y_0 - c_4*y_1))) - (c_6 + y_0)*(c_2*c_6*y_2*yx13*yx3 + yx14*(c_2*c_6*yx3 - c_3*(c_6 + y_0)**3)))/(c_6 + y_0)**4], [(c_2*c_6*y_2*yx16*yx5 + (c_6 + y_0)**4*(2*c_3*yx15 + yx10*(2*c_3*y_0 - 2*c_4*y_1 + c_4)) + (c_6 + y_0)**3*(-c_0*c_6*yx5 - c_0*y_0*yx5 + c_1*c_6*y_0*yx5 + c_1*y_0**2*yx5 + c_2*y_0*y_2*yx5 - c_2*y_0*yx6 - 2*c_3*c_6*y_0*yx10 - 2*c_3*y_0**2*yx10 + 2*c_4*c_6*y_1*yx10 + 2*c_4*y_0*y_1*yx10 - yx5*(c_2*y_0*y_2 - (c_0 - c_1*y_0)*(c_6 + y_0))) + (c_6 + y_0)**2*(c_2*c_6*yx5*yx8 - yx12*(c_2*c_6*y_2 + (c_1 + 2*c_4)*(c_6 + y_0)**2)) - (c_6 + y_0)*(c_2*c_6*yx14*yx5 + yx13*(c_2*c_6*y_2*yx5 - c_3*(c_6 + y_0)**3)))/(c_6 + y_0)**4], [(c_2*y_2*yx16*(c_6*(2*y_0 + 1) + 2*y_0**2 - 4*y_0*(c_6 + y_0) + 2*(c_6 + y_0)**2) + c_2*yx8*(c_6 + y_0)**2*(2*c_6*y_0 + c_6 - 2*y_0*(2*c_6 + y_0)) + (c_6 + y_0)**3*(c_0*c_6 + c_0*y_0 + c_1*c_6*y_0 + c_1*y_0**2 + c_2*y_0*y_2) - (c_6 + y_0)*(c_2*yx14*(c_6*(2*y_0 + 1) + 2*y_0**2 - 4*y_0*(c_6 + y_0) + 2*(c_6 + y_0)**2) + yx13*(2*c_1*(c_6 + y_0)**3 + c_2*c_6*y_2*(2*y_0 + 1) + 2*c_2*y_2*(y_0**2 - 2*y_0*(c_6 + y_0) + (c_6 + y_0)**2))))/(c_6 + y_0)**4], [(2*c_2*c_6*y_2*yx16*yx8*(c_6 + y_0)**3 + c_4*yx15*(c_6 + y_0)**3*(c_6**4 + 4*c_6**3*y_0 + 6*c_6**2*y_0**2 + 4*c_6*y_0**3 + y_0**4) - yx13*(2*c_2*c_6*y_2*yx8 + (c_6 + y_0)**3*(-c_4*y_1 + c_5*y_2))*(c_6**4 + 4*c_6**3*y_0 + 6*c_6**2*y_0**2 + 4*c_6*y_0**3 + y_0**4) - yx14*(2*c_2*y_2*(c_6 + y_0)**2 + c_2*(-2*c_6*y_0*y_2 + c_6*y_2 + 2*c_6*yx8 - 2*y_0**2*y_2) + (2*c_1 + c_5)*(c_6 + y_0)**3)*(c_6**4 + 4*c_6**3*y_0 + 6*c_6**2*y_0**2 + 4*c_6*y_0**3 + y_0**4) + (c_6 + y_0)**2*(c_6**4 + 4*c_6**3*y_0 + 6*c_6**2*y_0**2 + 4*c_6*y_0**3 + y_0**4)*(-2*c_0*c_6*yx8 - 2*c_0*y_0*yx8 + 2*c_1*c_6*y_0*yx8 + 2*c_1*y_0**2*yx8 + 2*c_2*y_0*y_2*yx8 + c_2*y_0*yx1 - c_4*c_6*y_1*yx13 - c_4*y_0*y_1*yx13 + c_5*c_6*y_2*yx13 + c_5*y_0*y_2*yx13) + (c_6 + y_0)*(c_2*yx9*(2*c_6*y_0 + c_6 - 2*y_0*(2*c_6 + y_0)) + yx8*(-2*c_2*c_6*y_0*y_2 + c_2*c_6*y_2 + 2*c_2*c_6*yx8 - 2*c_2*y_0**2*y_2 + (c_6 + y_0)**2*(2*c_0 - 2*c_1*y_0 + c_1)))*(c_6**4 + 4*c_6**3*y_0 + 6*c_6**2*y_0**2 + 4*c_6*y_0**3 + y_0**4))/((c_6 + y_0)**3*(c_6**4 + 4*c_6**3*y_0 + 6*c_6**2*y_0**2 + 4*c_6*y_0**3 + y_0**4))], [(c_1*c_6**4*yx10 - 2*c_1*c_6**4*yx15 + 4*c_1*c_6**3*y_0*yx10 - 8*c_1*c_6**3*y_0*yx15 + 6*c_1*c_6**2*y_0**2*yx10 - 12*c_1*c_6**2*y_0**2*yx15 + 4*c_1*c_6*y_0**3*yx10 - 8*c_1*c_6*y_0**3*yx15 + c_1*y_0**4*yx10 - 2*c_1*y_0**4*yx15 - 2*c_2*c_6**3*y_0*yx11 + c_2*c_6**3*y_0*yx3 + c_2*c_6**3*y_2*yx10 - 2*c_2*c_6**3*y_2*yx15 + 2*c_2*c_6**3*yx10*yx8 + c_2*c_6**3*yx11 - 6*c_2*c_6**2*y_0**2*yx11 + 3*c_2*c_6**2*y_0**2*yx3 + 2*c_2*c_6**2*y_0*y_2*yx10 - 4*c_2*c_6**2*y_0*y_2*yx15 + 4*c_2*c_6**2*y_0*yx10*yx8 + 2*c_2*c_6**2*y_0*yx11 - 2*c_2*c_6**2*y_2*yx10*yx13 - c_2*c_6**2*y_2*yx15 - 2*c_2*c_6**2*yx10*yx14 - 6*c_2*c_6*y_0**3*yx11 + 3*c_2*c_6*y_0**3*yx3 + c_2*c_6*y_0**2*y_2*yx10 - 2*c_2*c_6*y_0**2*y_2*yx15 + 2*c_2*c_6*y_0**2*yx10*yx8 + c_2*c_6*y_0**2*yx11 - 2*c_2*c_6*y_0*y_2*yx10*yx13 - c_2*c_6*y_0*y_2*yx15 - 2*c_2*c_6*y_0*yx10*yx14 + 2*c_2*c_6*y_2*yx10*yx16 - 2*c_2*y_0**4*yx11 + c_2*y_0**4*yx3 + c_3*c_6**4*yx16 + 4*c_3*c_6**3*y_0*yx16 + 6*c_3*c_6**2*y_0**2*yx16 + 4*c_3*c_6*y_0**3*yx16 + c_3*y_0**4*yx16 - c_4*c_6**4*yx15 - 4*c_4*c_6**3*y_0*yx15 - 6*c_4*c_6**2*y_0**2*yx15 - 4*c_4*c_6*y_0**3*yx15 - c_4*y_0**4*yx15)/(c_6**4 + 4*c_6**3*y_0 + 6*c_6**2*y_0**2 + 4*c_6*y_0**3 + y_0**4)], [(-c_2*yx14*(c_6 + y_0)**4*(c_6**2 + 2*c_6*y_0 + y_0**2)*(3*c_6**2*y_0 - 3*c_6**2 + 6*c_6*y_0**2 - 3*c_6*y_0 + 3*c_6*yx13 - c_6 + 3*y_0**3) + c_2*yx8*(c_6 + y_0)**4*(c_6**3 + 3*c_6**2*y_0 + 3*c_6*y_0**2 + y_0**3)*(3*c_6*y_0 + 3*c_6*yx13 - c_6 + 3*y_0**2) - yx13*(c_6 + y_0)*(c_6**2 + 2*c_6*y_0 + y_0**2)*(3*c_2*y_2*(c_6 + y_0)**2*(y_0 - 1) + c_2*y_2*(3*c_6*y_0 + 3*c_6*yx13 - c_6 + 3*y_0**2) + 3*(c_6 + y_0)**3*(-c_0 + c_1*y_0 - c_1))*(c_6**3 + 3*c_6**2*y_0 + 3*c_6*y_0**2 + y_0**3) - yx16*(c_6**2 + 2*c_6*y_0 + y_0**2)*(c_6**3 + 3*c_6**2*y_0 + 3*c_6*y_0**2 + y_0**3)*(3*c_1*(c_6 + y_0)**4 + 3*c_2*y_2*(c_6 + y_0)**3 + 3*c_2*y_2*(c_6 + y_0)**2*(-y_0 + 1) + c_2*y_2*(-3*c_6*y_0 - 3*c_6*yx13 + c_6 - 3*y_0**2)) + (c_6 + y_0)**3*(c_6**2 + 2*c_6*y_0 + y_0**2)*(c_6**3 + 3*c_6**2*y_0 + 3*c_6*y_0**2 + y_0**3)*(-3*c_0*c_6*yx13 + c_0*c_6 - 3*c_0*y_0*yx13 + c_0*y_0 + 3*c_1*c_6*y_0*yx13 - c_1*c_6*y_0 + 3*c_1*y_0**2*yx13 - c_1*y_0**2 + 3*c_2*y_0*y_2*yx13 - c_2*y_0*y_2))/((c_6 + y_0)**4*(c_6**2 + 2*c_6*y_0 + y_0**2)*(c_6**3 + 3*c_6**2*y_0 + 3*c_6*y_0**2 + y_0**3))]])

        problem = ODEProblem('MEA', ode_lhs_terms, right_hand_side, constants)

        simulation = Simulation(problem)
        timepoints = np.arange(0, 20.5, 0.5)

        parameters = [90, 0.002, 1.2, 1.1, 0.8, 0.96, 0.01]
        initial_conditions = [80, 40, 60]

        simulated_trajectories = simulation.simulate_system(parameters, initial_conditions, timepoints)
        answer_lookup = {trajectory.description: trajectory.values for trajectory in simulated_trajectories}

        # This is copy & paste from the model answer as well
        assert_array_almost_equal(answer_lookup[Moment(np.array([1, 0, 0]), symbol=y_0)],
                                  np.array([80.0, 91.2610806603, 102.686308216, 109.933251684, 110.94869635, 105.350207648, 94.0538240149, 78.9247701932, 62.3869294678, 47.001960274, 35.0615898206, 28.2435020082, 27.3706704079, 32.3183248969, 42.0889928739, 54.9985578025, 68.9536947619, 81.7868219838, 91.5901657401, 96.9959329891, 97.3627386877, 92.8421364264, 84.3236657923, 73.272168988, 61.4762424533,50.7637120227,42.7231560962,38.469283997, 38.4908216837, 42.6039654765, 50.0117466282, 59.4506901701, 69.4001577002,78.3243698773, 84.9069286181, 88.2422259139, 87.9566590434, 84.2435838803, 77.8103445963, 69.7494903693,61.3565372955]),
                                  decimal=2)
        assert_array_almost_equal(answer_lookup[Moment(np.array([0, 1, 0]), symbol=y_1)],
                                  np.array([40.0, 65.6770623435, 88.2517033254, 107.667075679, 122.502876961, 131.296471991, 133.22065575, 128.379472004, 117.838728384, 103.460946864, 87.6050572525, 72.7539295436, 61.1324254185, 54.3761228132, 53.3084553733, 57.8539387246, 67.0883391948, 79.4118924158, 92.8207267337, 105.228323011, 114.786903686, 120.158213069, 120.696036058, 116.519660015, 108.460834403, 97.9075142165, 86.5713502466, 76.2126043249, 68.3688586914, 64.1294568203, 63.9889146151, 67.7978718302, 74.8123983636, 83.8316854997, 93.3994744993, 102.036171999, 108.465098662, 111.798658105, 111.657310018, 108.206457547, 102.108817847]),
                                  decimal=2
                                  )
        assert_array_almost_equal(answer_lookup[Moment(np.array([0, 0, 1]), symbol=y_2)],
                                  np.array([60.0, 54.3421911216, 58.4333526621, 67.6229553127, 78.7343069688, 89.3188813562, 97.4939357634, 101.987525359, 102.212678494, 98.2878889685, 90.9725160657, 81.5172963056, 71.4503707325, 62.3306284628, 55.5071120261, 51.9222161344, 51.9866824269, 55.5390257081, 61.8937424341, 69.9671881798, 78.4588749996, 86.058450175, 91.6448538086, 94.4505773138, 94.1638155953, 90.9562404253, 85.4375235614, 78.5444651593, 71.3844585646, 65.0592711868, 60.4973013154, 58.319649156, 58.7585582684, 61.6384789746, 66.4196124678, 72.2951169645, 78.3248932497, 83.5843190466, 87.3043607498, 88.9822229399, 88.4469327849]),
                                  decimal=2
                                  )

        assert_array_almost_equal(answer_lookup[Moment(np.array([0, 0, 2]), symbol=yx1)],
                                  np.array([0.0, 28.5993791662, 45.7748086198, 66.8364968772, 97.0947506843, 135.324577478, 175.918470887, 212.31000862, 240.375722218, 260.107243953, 274.853957518, 288.844520386, 304.597957105, 321.741816721, 337.782452632, 350.164882734, 358.251394288, 363.97455251, 370.904035036, 382.310253159, 399.394352561, 420.697987766, 442.908583781, 462.53933381, 477.619668085, 488.305923485, 496.288307991, 503.522642122, 511.038572256, 518.536676537, 524.922116868, 529.325698636, 531.93458282, 534.1185569, 537.800325115, 544.462711439, 554.369432432, 566.432157247, 578.765442617, 589.579434648, 597.916484414]),
                                  decimal=1
                                  )

        assert_array_almost_equal(answer_lookup[Moment(np.array([0, 1, 1]), symbol=yx3)],
                                  np.array([0.0, -7.32141360954, -2.24188973022, 17.5084797878, 51.6590716616, 93.8424235009, 135.343040301, 170.125200595, 197.650834408, 222.064241424, 248.607059321, 279.68410952, 313.00822376, 342.862180452, 363.538354219, 372.774813724, 373.161139642, 370.851519565, 372.475113224, 382.015190231, 399.351371229, 421.094703512, 442.917627307, 462.011300298, 478.29846905, 493.678756873, 510.16056023, 528.132201684, 545.785549665, 560.06214574, 568.451828264, 570.517210113, 568.28443043, 565.318789843, 565.069258316, 569.403947687, 578.081337999, 589.33177882, 601.083092824, 612.064969993, 622.199907962]),
                                  decimal=1
                                  )

        assert_array_almost_equal(answer_lookup[Moment(np.array([0, 2, 0]), symbol=yx5)],
                                  np.array([0.0, 54.6787488077, 121.39466913, 208.452202663, 306.041394336, 398.631549654, 474.572074129, 531.509736428, 575.808018421, 617.14495972, 661.963602259, 709.76483484, 754.22517545, 788.111914024, 808.5780048, 819.577948708, 830.147055338, 849.814441155, 883.767675783, 930.493575276, 983.033174037, 1032.88857047, 1074.14663068, 1105.79288004, 1130.96518809, 1153.94697517, 1177.14567677, 1199.74031283, 1218.65119073, 1231.09707496, 1236.92907082, 1239.31220787, 1243.43726987, 1254.03297038, 1273.12032274, 1299.21090925, 1328.28308752, 1355.87861429, 1379.08417456, 1397.37788559, 1412.0651894]),
                                  decimal=1
                                  )
        assert_array_almost_equal(answer_lookup[Moment(np.array([1, 0, 1]), symbol=yx8)],
                                  np.array([0.0, -5.59381093653, -6.01373188616, 4.83142465175, 21.1739562401, 34.3051898654, 38.7010263034, 34.9542118715, 28.7051860602, 26.6784148891, 32.3988879223, 44.137987485, 56.1488971418, 62.265725446, 59.5304556748, 49.7465298949, 38.2170297585, 30.6823761487, 30.2597915223, 36.1015909812, 44.3693374321, 50.7517016884, 52.9062004222, 51.5803412395, 49.7348796376, 50.3804842011, 54.6420309538, 61.0901027356, 66.6539224015, 68.5142468372, 65.740746701, 59.7553546466, 53.4677067784, 49.6591327818, 49.5869839596, 52.5489591341, 56.5535678329, 59.6125884426, 60.8470250183, 60.800454067, 60.8665015286]),
                                  decimal=1
                                  )
        assert_array_almost_equal(answer_lookup[Moment(np.array([1, 1, 0]), symbol=yx10)],
                                  np.array([0.0, 22.0593723794, 75.299753943, 143.661956968, 211.305668627, 267.236954807, 309.659062883, 344.998381113, 382.251685516, 426.376615582, 474.764097865, 518.758765488, 549.121183033, 561.883088807, 560.849883932, 555.762517078, 557.376667829, 572.345843573, 600.585450753, 636.385181849, 672.254976881, 703.054272599, 728.086798434, 750.392217117, 773.65151975, 799.101577156, 824.414773703, 844.967491764, 856.71054717, 858.803882399, 854.365801154, 849.078496854, 848.542121578, 855.821742932, 870.477756188, 889.428965608, 908.966609026, 926.681878939, 942.243973437, 956.777665735, 971.416833834]),
                                  decimal=1
                                  )
        assert_array_almost_equal(answer_lookup[Moment(np.array([2, 0, 0]), symbol=yx13)],
                                  np.array([0.0, 81.5258461273, 167.954546114, 251.942490857, 324.828230888, 385.701673279, 441.363467087, 500.6915347, 567.946146057, 639.329536166, 704.771360708, 753.728199613, 781.558290863, 792.373922256, 796.613372847, 805.836927684, 827.489463493, 862.171932346, 904.702331823, 948.012305446, 987.346780966, 1022.30991727, 1055.94217358, 1091.74005065, 1130.39137271, 1168.69093753, 1201.08852899, 1222.77926963, 1232.58262231, 1233.76888086, 1232.45317329, 1234.77059327, 1244.34495423, 1261.24208422, 1282.80462044, 1305.66948076, 1327.68715245, 1348.67228282, 1369.71661676, 1391.6992575, 1414.03174622]),
                                  decimal=1
                                  )

        

class TestSimulateWithSensitivities(unittest.TestCase):


    def test_model_in_paper(self):
        """
        Given the model in the Ale et. al Paper, and the initial parameters,
        the simulation with sensitivities result should be similar to the one described in paper, within minimal margin
        of error.
        """
        parameters = [1.66e-3, 0.2]
        initial_conditions = [301, 0]
        timepoints = np.arange(0, 20, 0.1)

        problem = means.approximation.ODEProblem('MNA',
                                                 [Moment([1, 0], 'x_1'),
                                                  Moment([0, 1], 'x_2'),
                                                  Moment([0, 2], 'yx1'),
                                                  Moment([1, 1], 'yx2'),
                                                  Moment([2, 0], 'yx3')],
                                                 to_sympy_matrix(['-2*k_1*x_1*(x_1 - 1) - 2*k_1*yx3 + 2*k_2*x_2',
                                                                  'k_1*x_1*(x_1 - 1) + k_1*yx3 - k_2*x_2',

                                                                  'k_1*x_1**2 - k_1*x_1 + 2*k_1*yx2*(2*x_1 - 1) '
                                                                  '+ k_1*yx3 + k_2*x_2 - 2*k_2*yx1',

                                                                  '-2*k_1*x_1**2 + 2*k_1*x_1 + k_1*yx3*(2*x_1 - 3) '
                                                                  '- 2*k_2*x_2 + 2*k_2*yx1 - yx2*(4*k_1*x_1 '
                                                                  '- 2*k_1 + k_2)',

                                                                  '4*k_1*x_1**2 - 4*k_1*x_1 - 8*k_1*yx3*(x_1 - 1)'
                                                                  ' + 4*k_2*x_2 + 4*k_2*yx2'
                                                                  ]),
                                                 ['k_1', 'k_2']
                                                 )

        simulation = means.simulation.SimulationWithSensitivities(problem)
        trajectories = simulation.simulate_system(parameters, initial_conditions, timepoints)

        answers = {}

        # Trajectory value, sensitivity wrt k_1, sensitivity wrt k_2
        answers[Moment([1, 0], 'x_1')] = (107.948953772, -25415.3565093, 210.946558295)
        answers[Moment([0, 1], 'x_2')] = (96.5255231141, 12707.6782547, -105.473279147)

        seen_answers = set()
        for trajectory in trajectories:
            # There should be one sensitivity trajectory for each parameter
            self.assertEqual(len(trajectory.sensitivity_data), len(parameters))

            # Check the ones we have answers for
            answer = None
            try:
                answer = answers[trajectory.description]
            except KeyError:
                continue

            seen_answers.add(trajectory.description)

            self.assertAlmostEqual(answer[0], trajectory.values[-1], delta=1e-6)
            self.assertAlmostEqual(answer[1], trajectory.sensitivity_data[0].values[-1], delta=1e-6)
            self.assertAlmostEqual(answer[2], trajectory.sensitivity_data[1].values[-1], delta=1e-6)

        self.assertEqual(len(seen_answers), len(answers), msg='Some of the trajectories for moments were not returned')