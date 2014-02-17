import unittest
import means
from means.util.sympyhelpers import to_sympy_matrix
from means.approximation.ode_problem import ODEProblem, ODETermBase, Moment, VarianceTerm
from means.simulation import Simulation
from numpy.testing import assert_array_almost_equal
import numpy as np

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