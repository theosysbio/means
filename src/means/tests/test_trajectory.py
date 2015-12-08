import unittest
import numpy as np
from means.simulation import Trajectory, TrajectoryWithSensitivityData, TrajectoryCollection
from means.core import Moment
import os
import tempfile

class TestTrajectory(unittest.TestCase):

    def test_equality_treats_equal_things_as_equal(self):
        """
        Given two Trajectories that were equal, they should be comparable with ==.
        """

        t1 = Trajectory([1, 2, 3], [3, 2, 1], Moment([1], symbol='description'))
        t2 = Trajectory([1, 2, 3], [3, 2, 1], Moment([1], symbol='description'))

        self.assertEqual(t1, t2)

    def test_different_timepoints_make_trajectories_different(self):
        """
        Given two Trajectories that differ only by timepoints, they should be treated as different
        """

        t1 = Trajectory([1, 2, 3], [3, 2, 1], Moment([1], symbol='description'))
        t2 = Trajectory([0, 1, 2], [3, 2, 1], Moment([1], symbol='description'))

        self.assertNotEqual(t1, t2)

    def test_different_values_make_trajectories_different(self):
        """
        Given two Trajectories that differ only by values, they should be treated as different
        """

        t1 = Trajectory([1, 2, 3], [3, 2, 1], Moment([1], symbol='description'))
        t2 = Trajectory([1, 2, 3], [4, 2, 1], Moment([1], symbol='description'))

        self.assertNotEqual(t1, t2)

    def test_different_descriptions_make_trajectories_different(self):
        """
        Given two Trajectories that differ only by values, they should be treated as different
        """

        t1 = Trajectory([1, 2, 3], [3, 2, 1], Moment([1], symbol='description'))
        t2 = Trajectory([1, 2, 3], [3, 2, 1], Moment([1], symbol='description'))

        self.assertNotEqual(t1, t2)


class TestTrajectoryWithSensitivityData(unittest.TestCase):

    def test_equality_treats_equal_things_as_equal(self):
        """
        Given two Trajectories that were equal, they should be comparable with ==.
        """
        t_sensitivity_1 = Trajectory([1,2,3], [3, 2, 1], Moment([1], symbol='description'))
        t_sensitivity_2 = Trajectory([1,2, 3], [-5, -9, -1], Moment([1], symbol='description'))

        t1 = TrajectoryWithSensitivityData([1, 2, 3], [3, 2, 1], Moment([1], symbol='description'), [t_sensitivity_1, t_sensitivity_2])
        t2 = TrajectoryWithSensitivityData([1, 2, 3], [3, 2, 1], Moment([1], symbol='description'), [t_sensitivity_1, t_sensitivity_2])

        self.assertEqual(t1, t2)

    def test_different_timepoints_make_trajectories_different(self):
        """
        Given two TrajectoriesWithSensitivityData that differ only by sensitivity data
        they should be reported as different
        """

        t_sensitivity_1 = Trajectory([1,2,3], [3, 2, 1], Moment([1], symbol='sensitivity1'))
        t_sensitivity_2 = Trajectory([1,2, 3], [-5, -9, -1], Moment([1], symbol='sensitivity2'))
        t_sensitivity_3 = Trajectory([1,2,3], [-5, -9, -100], Moment([1], symbol='sensitivity2'))

        t1 = TrajectoryWithSensitivityData([1, 2, 3], [3, 2, 1], Moment([1], symbol='description'), [t_sensitivity_1, t_sensitivity_2])
        t2 = TrajectoryWithSensitivityData([1, 2, 3], [3, 2, 1], Moment([1], symbol='description'), [t_sensitivity_1, t_sensitivity_3])

        self.assertNotEqual(t1, t2)


class TestTrajectoriesToCSV(unittest.TestCase):

    def test_single_traj_to_file(self):

        trajectory = Trajectory([1, 2, 3, 4, 5, 6], [3, 2, 1,5, 2, 4], Moment([1], symbol='description'))
        file = tempfile.mktemp(suffix=".csv")
        try:
            with open(file,"w") as out:
                trajectory.to_csv(out)
        finally:
            os.unlink(file)

    def test_traj_collection_to_file(self):

        tr2 = Trajectory([1, 2, 3, 4, 5, 6], [3, 2, 1, 5, 2, 4], Moment([1], symbol='y_1'))
        tr1 = Trajectory([1, 2, 3, 4, 5, 6], [3, 2, 1, 5, 2, 4], Moment([1], symbol='y_2'))
        tc = TrajectoryCollection([tr1,tr2])

        file = tempfile.mktemp(suffix=".csv")
        try:
            with open(file,"w") as out:
                tc.to_csv(out)
        finally:
            os.unlink(file)



