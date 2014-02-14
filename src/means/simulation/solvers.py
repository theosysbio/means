from collections import namedtuple
from assimulo.problem import Explicit_Problem
import numpy as np
from means.simulation.trajectory import Trajectory

NP_FLOATING_POINT_PRECISION = np.double

RTOL = 1e-4
ATOL = 1e-4

_Solver = namedtuple('_Solver', ['name', 'supports_sensitivity'])

class SolverBase(object):

    short_name = None  # This name is used when the solver is initialised by string parameter
    supports_sensitivity = None  # Whether solver supports sensitivity calculations or not

    _parameters = None
    _initial_conditions = None
    _problem = None
    _starting_time = None
    _options = None
    _solver = None
    _solver_exception_class = None

    def simulate(self, timepoints):
        solver = self._solver
        last_timepoint = timepoints[-1]

        try:
            simulated_timepoints, simulated_values = solver.simulate(last_timepoint, ncp_list=timepoints)
        except self._solver_exception_class as e:
            # The exceptions thrown by solvers are usually hiding the real cause, try to see if it is
            # our right_hand_side_as_function that is broken first
            try:
                self._problem_right_hand_side(self._initial_conditions, self._parameters)
            except:
                # If it is broken, throw that exception instead
                raise
            else:
                # If it is not, re-raise the original exception
                raise e

        return self._postprocess_results(simulated_timepoints, simulated_values, solver)

    def _postprocess_results(self, simulated_timepoints, simulated_values, solver):
        pass

    def _initialise_model(self):
        raise NotImplementedError

    def _set_options_to_solver(self):
        # We want to silence all solvers by default, don't we?
        self._solver.verbosity = self._options.pop('verbosity', 50)

         # Set the remaining attributes
        for attribute, value in self._options:
            setattr(self._solver, attribute, value)

    def __init__(self, problem, parameters, initial_conditions, starting_time=0.0, **options):

        parameters = np.array(parameters, dtype=NP_FLOATING_POINT_PRECISION)
        initial_conditions = np.array(initial_conditions, dtype=NP_FLOATING_POINT_PRECISION)
        assert(parameters.shape == (len(problem.constants),))
        assert(initial_conditions.shape == (problem.number_of_equations,))

        self._parameters = parameters
        self._initial_conditions = initial_conditions
        self._starting_time = float(starting_time)
        self._problem = problem
        self._options = options

        self._initialise_model()
        self._solver = self._get_solver_instance()
        self._set_options_to_solver()

    def _get_solver_instance(self):
        raise NotImplementedError

class NonparametricSolverBase(SolverBase):
    _model = None

    def _initialise_model(self):
        rhs = self._problem.right_hand_side_as_function
        parameters = self._parameters
        initial_conditions = self._initial_conditions
        initial_timepoint = self._starting_time

        model = Explicit_Problem(lambda t, x: rhs(x, parameters),
                                 initial_conditions, initial_timepoint)

        self._model = model

    def _postprocess_results(self, simulated_timepoints, simulated_values, solver):
        trajectories = []
        descriptions = self._problem.ordered_descriptions

        number_of_timepoints, number_of_simulated_values = simulated_values.shape

        assert(len(descriptions) == number_of_simulated_values)
        assert(len(simulated_timepoints) == number_of_timepoints)

        for description, simulated_value_column in zip(descriptions, simulated_values.T):
            trajectories.append(Trajectory(simulated_timepoints, simulated_value_column, description))

        return trajectories

class SensitivitySolverBase(NonparametricSolverBase):

    def _initialise_model(self):
        rhs = self._problem.right_hand_side_as_function
        parameters = self._parameters
        initial_conditions = self._initial_conditions
        initial_timepoint = self._starting_time

        # Solvers with sensitivity support should be able to accept parameters
        # into rhs function directly
        model = Explicit_Problem(lambda t, x, p: rhs(x, p),
                                 initial_conditions, initial_timepoint)

        model.p0 = np.array(parameters)
        self._model = model


class Dopri5Solver(NonparametricSolverBase):

    def _get_solver_instance(self):
        from assimulo.solvers.runge_kutta import Dopri5
        return Dopri5(self._model)

class CVodeSolver(NonparametricSolverBase):

    def _get_solver_instance(self):
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




