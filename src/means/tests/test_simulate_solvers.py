from __future__ import absolute_import, print_function

import unittest

import numpy as np
import sympy

from means.core import ODETermBase
from means.simulation import Trajectory, TrajectoryWithSensitivityData, SensitivityTerm
from means.simulation.solvers import _wrap_results_to_trajectories, _add_sensitivity_data_to_trajectories


class TestSolverBase(unittest.TestCase):

    def _test_trajectory_wrapping_works_correctly(self):
        """
        Given a set of returned values from a solver, _results_to_trajectories should be able to
        wrap those results into :class:`~means.simulation.trajectory.Trajectory` objects nicely.
        """

        descriptions = [ODETermBase('x'), ODETermBase('y')]

        simulated_timepoints = [1,2,3]
        simulated_values = np.array([[-3, 1], [-2, 2], [-1, 3]])

        trajectory1, trajectory2 = _wrap_results_to_trajectories(simulated_timepoints, simulated_values, descriptions)

        correct_trajectory1 = Trajectory(simulated_timepoints, [-3, -2, -1], descriptions[0])
        correct_trajectory2 = Trajectory(simulated_timepoints, [1, 2, 3], descriptions[1])

        self.assertEqual(correct_trajectory1, trajectory1)
        self.assertEqual(correct_trajectory2, trajectory2)

class TestSensitivitySolverBase(unittest.TestCase):

    def _test_trajectory_with_sensitivity_wrapping_works_correctly(self):
        """
        Given a set of already pre-processed trajectories, results from sensitivity calculations,
        and the parameters of model, these should be wrapped nicely into TrajectoryWithSensitivityData objects
        """
        time_points = [1, 2, 3]

        t0 = Trajectory(time_points, [-1, 2, -3], ODETermBase('x'))
        t1 = Trajectory(time_points, [-1, -2, -3], ODETermBase('y'))

        parameters = sympy.symbols(['p', 'z'])

        raw_sensitivity_results = np.array([[[np.nan, np.nan], [1, 2], [3, 4]],
                                            [[np.nan, np.nan], [-1, -3], [-2, -4]]])

        actual_t1, actual_t2 = _add_sensitivity_data_to_trajectories([t0, t1], raw_sensitivity_results, parameters)

        sensitivity_data_1 = [Trajectory(time_points, [np.nan, 1, 3], SensitivityTerm(t0.description, parameters[0])),
                              Trajectory(time_points, [np.nan, 2, 4], SensitivityTerm(t1.description, parameters[1]))]
        expected_t1 = TrajectoryWithSensitivityData.from_trajectory(t0, sensitivity_data_1)

        sensitivity_data_2 = [Trajectory(time_points, [np.nan, -1, -2], SensitivityTerm(t0.description, parameters[0])),
                              Trajectory(time_points, [np.nan, -3, -4], SensitivityTerm(t1.description, parameters[1]))]
        expected_t2 = TrajectoryWithSensitivityData.from_trajectory(t0, sensitivity_data_2)

        self.assertEqual(actual_t1, expected_t1)
        self.assertEqual(actual_t2, expected_t2)
