import unittest
import numpy as np
from means.approximation.ode_problem import ODETermBase
from means.simulation import Trajectory
from means.simulation.solvers import _wrap_results_to_trajectories

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
