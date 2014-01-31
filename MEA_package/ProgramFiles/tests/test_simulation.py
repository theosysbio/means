import unittest
from ode_problem import ODEProblem
from simulate import Simulation
import numpy as np


class StubODEProblemWithConstantRHSAsFunction(ODEProblem):

    def __init__(self):
        pass

    @property
    def rhs_as_function(self):
        return lambda c1, c2, y1, y2, y3: np.array([c1 - c2 + y3*2 - y2, c1+c2, y1+c1])

class TestSimulation(unittest.TestCase):

    def test_simulation_rhs_function_factory(self):
        """
        Given a simulation class for StubODEProblem instance, the function _rhs_function_factory should return a function
        that takes a vector of values for all non-constant variables in the function and returns their values post the
        function calculation.
        """
        s = Simulation(StubODEProblemWithConstantRHSAsFunction())

        rhs_as_function = s._rhs_function_factory(np.array([1, 2]))
        initial_values = np.array([3, 4, 5])
        timepoint = 15  # let's make this 15, why not?
        # Computed from the rhs_as_function in StubODEProblemWithConstantRHSAsFunction
        correct_answer = np.array([5, 3, 4])
        self.assertListEqual(list(rhs_as_function(timepoint, initial_values)), list(correct_answer))