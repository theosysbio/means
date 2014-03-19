import unittest
import sympy
from means.simulation import Trajectory
from means import  Moment
from means import SSASimulation
from means import StochasticProblem
from means.examples import MODEL_LOTKA_VOLTERRA

class TestSSA(unittest.TestCase):

    def test_moment_from_traj(self):
        dummy = StochasticProblem(MODEL_LOTKA_VOLTERRA)
        ssa = SSASimulation(dummy, 1)

        all_trajectories = [
            [
                Trajectory([0, 1, 2], [3, 4, 4], Moment([0,1],"a")),
                Trajectory([0, 1, 2], [7, 9, 10], Moment([1,0],"b"))
            ],
            [
                Trajectory([0, 1, 2], [1, 2, 5], Moment([0,1],"a")),
                Trajectory([0, 1, 2], [2, 3, 3], Moment([1,0],"b"))
            ],
            [
                Trajectory([0, 1, 2], [13,7,5], Moment([0,1],"a")),
                Trajectory([0, 1, 2], [22,3,1], Moment([1,0],"b"))
            ],
            [
                Trajectory([0, 1, 2], [9,21, 3], Moment([0,1],"a")),
                Trajectory([0, 1, 2], [8, 7, 4], Moment([1,0],"b"))
            ]
        ]

        mean_trajectories = [sum(trajs)/float(len(trajs)) for trajs in zip(*all_trajectories)]

        variance_a_result = ssa._compute_one_moment(all_trajectories, mean_trajectories,
                                                    Moment([2,0],sympy.Symbol("V_a")))
        covar_result= ssa._compute_one_moment(all_trajectories, mean_trajectories,
                                              Moment([1,1],sympy.Symbol("Cov_a_b")))

        variance_a_expected = Trajectory([0.0, 1.0, 2.0],[22.75, 55.25, 0.6875],
                                         Moment([2, 0], sympy.Symbol("V_a")))

        covar_expected = Trajectory([0.0, 1.0, 2.0], [31.875, 5.75, -1.125],
                                    Moment([1, 1], sympy.Symbol("Cov_a_b")))


        self.assertEqual(variance_a_result , variance_a_expected)
        self.assertEqual(covar_result, covar_expected)
