from collections import namedtuple
from assimulo.problem import Explicit_Problem
import numpy as np
from means.simulation.trajectory import Trajectory
from means.util.decorators import memoised_property

NP_FLOATING_POINT_PRECISION = np.double

RTOL = 1e-4
ATOL = 1e-4

_Solver = namedtuple('_Solver', ['name', 'supports_sensitivity'])

def _set_kwargs_as_attributes(instance, **kwargs):
    for attribute, value in kwargs.iteritems():
        setattr(instance, attribute, value)
    return instance

def _wrap_results_to_trajectories(simulated_timepoints, simulated_values, descriptions):
    number_of_timepoints, number_of_simulated_values = simulated_values.shape

    assert(len(descriptions) == number_of_simulated_values)
    assert(len(simulated_timepoints) == number_of_timepoints)

    # Wrap results to trajectories
    trajectories = []
    for description, simulated_value_column in zip(descriptions, simulated_values.T):
        trajectories.append(Trajectory(simulated_timepoints, simulated_value_column, description))

    return trajectories


class SolverBase(object):
    """
    This acts as a base class for ODE solvers used in `means`.
    It wraps around the solvers available in :module:`assimulo` package, and provides some basic functionality
    that allows solvers be used with `means` objects.
    """

    _parameters = None
    _initial_conditions = None
    _problem = None
    _starting_time = None
    _options = None

    def __init__(self, problem, parameters, initial_conditions, starting_time=0.0, **options):
        """

        :param problem: Problem to simulate
        :type problem: :class:`~means.approximation.ODEProblem`
        :param parameters: Parameters of the solver. One entry for each constant in `problem`
        :type parameters: :class:`iterable`
        :param initial_conditions: Initial conditions of the system. One for each of the equations.
                                   Assumed to be zero, if not specified
        :type initial_conditions: :class:`iterable`
        :param starting_time: Starting time for the solver, defaults to 0.0
        :type starting_time: float
        :param options: Options to be passed to the specific instance of the solver.
        """
        parameters = np.array(parameters, dtype=NP_FLOATING_POINT_PRECISION)
        initial_conditions = np.array(initial_conditions, dtype=NP_FLOATING_POINT_PRECISION)
        assert(parameters.shape == (len(problem.constants),))
        assert(initial_conditions.shape == (problem.number_of_equations,))

        self._parameters = parameters
        self._initial_conditions = initial_conditions
        self._starting_time = float(starting_time)
        self._problem = problem
        self._options = options

    def simulate(self, timepoints):
        """
        Simulate initialised solver for the specified timepoints

        :param timepoints: timepoints that will be returned from simulation
        :return: a list of trajectories for each of the equations in the problem.
        """
        solver = self._solver
        last_timepoint = timepoints[-1]

        try:
            simulated_timepoints, simulated_values = solver.simulate(last_timepoint, ncp_list=timepoints)
        except self._solver_exception_class as e:
            # The exceptions thrown by solvers are usually hiding the real cause, try to see if it is
            # our right_hand_side_as_function that is broken first
            try:
                self._problem.right_hand_side(self._initial_conditions, self._parameters)
            except:
                # If it is broken, throw that exception instead
                raise
            else:
                # If it is not, re-raise the original exception
                raise e

        return self._results_to_trajectories(simulated_timepoints, simulated_values)

    def _default_solver_instance(self):
        raise NotImplementedError

    @property
    def _solver_exception_class(self):
        """
        Property That would return the exception class thrown by a specific solver the subclases can override.
        """
        return None

    @memoised_property
    def _solver(self):
        solver = self._default_solver_instance()
        verbosity = self._options.pop('verbosity', 50)
        return _set_kwargs_as_attributes(solver, verbosity=verbosity, **self._options)

    @memoised_property
    def _model(self):
        rhs = self._problem.right_hand_side_as_function
        parameters = self._parameters
        initial_conditions = self._initial_conditions
        initial_timepoint = self._starting_time

        model = Explicit_Problem(lambda t, x: rhs(x, parameters),
                                 initial_conditions, initial_timepoint)

        return model

    def _results_to_trajectories(self, simulated_timepoints, simulated_values):
        """
        Convert the resulting results into a list of trajectories

        :param simulated_timepoints: timepoints output from a solver
        :param simulated_values: values returned by the solver
        :return:
        """

        descriptions = self._problem.ordered_descriptions

        return _wrap_results_to_trajectories(simulated_timepoints, simulated_values, descriptions)

class SensitivitySolverBase(SolverBase):

    def _model(self):
        rhs = self._problem.right_hand_side_as_function
        parameters = self._parameters
        initial_conditions = self._initial_conditions
        initial_timepoint = self._starting_time

        # Solvers with sensitivity support should be able to accept parameters
        # into rhs function directly
        model = Explicit_Problem(lambda t, x, p: rhs(x, p),
                                 initial_conditions, initial_timepoint)

        model.p0 = np.array(parameters)
        return model


class Dopri5Solver(SolverBase):

    def _default_solver_instance(self):
        from assimulo.solvers.runge_kutta import Dopri5
        return Dopri5(self._model)

class CVodeSolver(SolverBase):

    @property
    def _solver_exception_class(self):
        from assimulo.solvers.sundials import CVodeError
        return CVodeError

    def _default_solver_instance(self):
        from assimulo.solvers.sundials import CVode

        solver = CVode(self._model)

        options = self._options

        solver.iter = options.pop('iter', 'Newton')
        solver.discr = options.pop('discr', 'BDF')
        solver.atol = options.pop('atol', ATOL)
        solver.rtol = options.pop('rtol', RTOL)
        solver.linear_solver = options.pop('linear_solver', 'dense')

        if 'usesens' in options:
            # TODO: Change this with regard to how Simulation CLass changes
            raise AttributeError('Cannot set \'usesens\' parameter. Use Simulation or SimulationWithSensitivities for '
                                 'sensitivity calculations')

        # It is necessary to set usesens to false here as we are non-parametric here
        solver.usesens = False

        return solver




