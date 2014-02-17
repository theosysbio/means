import unittest
import numpy as np
from means.simulation import Trajectory, TrajectoryWithSensitivityData


class TestTrajectory(unittest.TestCase):

    def test_equality_treats_equal_things_as_equal(self):
        """
        Given two Trajectories that were equal, they should be comparable with ==.
        """

        t1 = Trajectory([1, 2, 3], [3, 2, 1], 'description')
        t2 = Trajectory([1, 2, 3], [3, 2, 1], 'description')

        self.assertEqual(t1, t2)

    def test_different_timepoints_make_trajectories_different(self):
        """
        Given two Trajectories that differ only by timepoints, they should be treated as different
        """

        t1 = Trajectory([1, 2, 3], [3, 2, 1], 'description')
        t2 = Trajectory([0, 1, 2], [3, 2, 1], 'description')

        self.assertNotEqual(t1, t2)

    def test_different_values_make_trajectories_different(self):
        """
        Given two Trajectories that differ only by values, they should be treated as different
        """

        t1 = Trajectory([1, 2, 3], [3, 2, 1], 'description')
        t2 = Trajectory([1, 2, 3], [4, 2, 1], 'description')

        self.assertNotEqual(t1, t2)

    def test_different_descriptions_make_trajectories_different(self):
        """
        Given two Trajectories that differ only by values, they should be treated as different
        """

        t1 = Trajectory([1, 2, 3], [3, 2, 1], 'description1')
        t2 = Trajectory([1, 2, 3], [3, 2, 1], 'description2')

        self.assertNotEqual(t1, t2)


class TestTrajectoryWithSensitivityData(unittest.TestCase):

    def test_equality_treats_equal_things_as_equal(self):
        """
        Given two Trajectories that were equal, they should be comparable with ==.
        """
        t_sensitivity_1 = Trajectory([1,2,3], [3, 2, 1], 'sensitivity1')
        t_sensitivity_2 = Trajectory([1,2, 3], [-5, -9, -1], 'sensitivity2')

        t1 = TrajectoryWithSensitivityData([1, 2, 3], [3, 2, 1], 'description', [t_sensitivity_1, t_sensitivity_2])
        t2 = TrajectoryWithSensitivityData([1, 2, 3], [3, 2, 1], 'description', [t_sensitivity_1, t_sensitivity_2])

        self.assertEqual(t1, t2)

    def test_different_timepoints_make_trajectories_different(self):
        """
        Given two TrajectoriesWithSensitivityData that differ only by sensitivity data
        they should be reported as different
        """

        t_sensitivity_1 = Trajectory([1,2,3], [3, 2, 1], 'sensitivity1')
        t_sensitivity_2 = Trajectory([1,2, 3], [-5, -9, -1], 'sensitivity2')
        t_sensitivity_3 = Trajectory([1,2,3], [-5, -9, -100], 'sensitivity2')

        t1 = TrajectoryWithSensitivityData([1, 2, 3], [3, 2, 1], 'description', [t_sensitivity_1, t_sensitivity_2])
        t2 = TrajectoryWithSensitivityData([1, 2, 3], [3, 2, 1], 'description', [t_sensitivity_1, t_sensitivity_3])

        self.assertNotEqual(t1, t2)
