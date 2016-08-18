from __future__ import absolute_import, print_function
"""
Solvers
-------

This part of the package provides wrappers around Assimulo solvers.
"""
import sys
import inspect

from assimulo.problem import Explicit_Problem
import numpy as np

from means.simulation import SensitivityTerm
from means.simulation.trajectory import Trajectory, TrajectoryWithSensitivityData
from means.util.memoisation import memoised_property, MemoisableObject
from means.util.sympyhelpers import to_one_dim_array

NP_FLOATING_POINT_PRECISION = np.double

#-- Easy initialisation utilities -------------------------------------------------------------

class UniqueNameInitialisationMixin(object):

    @classmethod
    def unique_name(self):
        return NotImplemented

class SolverException(Exception):

    __base_exception_class = None
    __base_exception_kwargs = None

    def __init__(self, message, base_exception=None):
        if base_exception is not None:
            if message is None:
                message = ''
            # We need to take message argument as otherwise SolverException is unpickleable
            message += '{0.__class__.__name__}: {0!s}'.format(base_exception)

        super(SolverException, self).__init__(message)

        # CVodeError does not serialise well, so let's store it as a set of arguments and create the base exception
        # on the fly, rather than storing the actual object
        if base_exception is not None:
            self.__base_exception_class = base_exception.__class__
            self.__base_exception_kwargs = base_exception.__dict__.copy()

    @property
    def base_exception(self):
        if self.__base_exception_class is not None:
            return self.__base_exception_class(**self.__base_exception_kwargs)


    def __eq__(self, other):
        return isinstance(other, self.__class__) and  \
               self.message == other.message and self.__base_exception_class == other.__base_exception_class  and \
               self.__base_exception_kwargs == other.__base_exception_kwargs


def available_solvers(with_sensitivity_support=False):
    members = inspect.getmembers(sys.modules[__name__])

    initialisable_solvers = {}
    # Some metaprogramming here: look for all classes at this module that are subclasses of
    # `UniqueNameInitialisationMixin`. Compile a dictionary of these
    for name, object in members:
        if inspect.isclass(object) and issubclass(object, SolverBase) \
                and issubclass(object, UniqueNameInitialisationMixin) \
                and object != UniqueNameInitialisationMixin:

            if with_sensitivity_support and not issubclass(object, SensitivitySolverBase):
                # If we need sensitivity support, skip all non-sensitivity solvers
                continue
            elif not with_sensitivity_support and issubclass(object, SensitivitySolverBase):
                # If we don't need sensitivity support, skip all solvers with sensitivity support
                continue

            assert(object.unique_name not in initialisable_solvers)
            initialisable_solvers[object.unique_name().lower()] = object

    return initialisable_solvers

#-- Exception handling utilities -----------------------------------------------------------


def parse_flag(exception_message):
    """
    Parse the flag from the solver exception.
    e.g.

    >>> parse_flag("Exception: Dopri5 failed with flag -3")
    -3

    :param exception_message: message from the exception
    :type exception_message: str
    :return: flag id
    :rtype: int
    """
    import re
    match = re.match('.* failed with flag (-\d+)', exception_message)
    try:
        return int(match.group(1))
    except Exception:
        return None

#-- Base solver functionality ---------------------------------------------------------------

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


class SolverBase(MemoisableObject):
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
        parameters = to_one_dim_array(parameters, dtype=NP_FLOATING_POINT_PRECISION)
        initial_conditions = to_one_dim_array(initial_conditions, dtype=NP_FLOATING_POINT_PRECISION)

        assert(parameters.shape == (len(problem.parameters),))
        assert(initial_conditions.shape[0] == problem.number_of_equations)

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

        except (Exception, self._solver_exception_class) as e:
            # The exceptions thrown by solvers are usually hiding the real cause, try to see if it is
            # our right_hand_side_as_function that is broken first
            try:
                self._problem.right_hand_side_as_function(self._initial_conditions, self._parameters)
            except:
                # If it is broken, throw that exception instead
                raise
            else:
                # If it is not, handle the original exception
                self._handle_solver_exception(e)

        trajectories =  self._results_to_trajectories(simulated_timepoints, simulated_values)

        return trajectories

    def _handle_solver_exception(self, solver_exception):
        """
        This function handles any exceptions that occurred in the solver and have been proven not to be
        related to our right_hand_side function.
        Subclasses can override it.

        :param solver_exception: the exception raised by the solver
        :type solver_exception: Exception
        """
        # By default just re-raise it with our wrapper
        raise SolverException(None, solver_exception)

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
    def _assimulo_problem(self):
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

        descriptions = self._problem.left_hand_side_descriptors

        return _wrap_results_to_trajectories(simulated_timepoints, simulated_values, descriptions)


class CVodeMixin(UniqueNameInitialisationMixin, object):

    @classmethod
    def unique_name(cls):
        return 'cvode'

    @property
    def _solver_exception_class(self):
        from assimulo.solvers.sundials import CVodeError
        return CVodeError

    def _cvode_instance(self, model, options):
        from assimulo.solvers.sundials import CVode

        solver = CVode(model)

        if 'usesens' in options:
            raise AttributeError('Cannot set \'usesens\' parameter. Use Simulation or SimulationWithSensitivities for '
                                 'sensitivity calculations')

        return solver

class CVodeSolver(SolverBase, CVodeMixin):

    def _default_solver_instance(self):
        solver = self._cvode_instance(self._assimulo_problem, self._options)
        # It is necessary to set usesens to false here as we are non-parametric here
        solver.usesens = False
        return solver


class ODE15sMixin(CVodeMixin):
    """
    A CVODE solver that mimicks the parameters used in `ode15s`_ solver in MATLAB.

    The different parameters that are set differently by default are:

    ``discr``
        Set to ``'BDF'`` by default
    ``atol``
        Set to ``1e-6``
    ``rtol``
        Set to ``1e-3``

    .. _`ode15s`: http://www.mathworks.ch/ch/help/matlab/ref/ode15s.html
    """

    ATOL = 1e-6
    RTOL = 1e-3
    MINH = 5.684342e-14

    @classmethod
    def unique_name(cls):
        return 'ode15s'

    def _cvode_instance(self, model, options):
        solver = super(ODE15sMixin, self)._cvode_instance(model, options)

        # BDF method below makes it a key similarity to the ode15s
        solver.discr = options.pop('discr', 'BDF')
        solver.atol = options.pop('atol', self.ATOL)
        solver.rtol = options.pop('rtol', self.RTOL)
        solver.maxord = options.pop('maxord', 5)
        # If minh is not set, CVODE would try to continue the simulation, issuing a warning
        # We set it here so this simulation fails.
        solver.minh = options.pop('minh', self.MINH)

        return solver

class ODE15sLikeSolver(SolverBase, ODE15sMixin):

    def _default_solver_instance(self):
        solver = self._cvode_instance(self._assimulo_problem, self._options)
        # It is necessary to set usesens to false here as we are non-parametric here
        solver.usesens = False
        return solver

class Dopri5Solver(SolverBase, UniqueNameInitialisationMixin):

    def _default_solver_instance(self):
        from assimulo.solvers.runge_kutta import Dopri5
        return Dopri5(self._assimulo_problem)

    @classmethod
    def unique_name(self):
        return 'dopri5'

    def _handle_solver_exception(self, solver_exception):
        # Let's try and parse the exception flag, to add some helpful info
        flag = parse_flag(solver_exception.message)

        FLAG_DOCUMENTATION = {-1: 'Input is not consistent',
                              -2: 'Larger NMAX is needed',
                              -3: 'Step size becomes too small',
                              -4: 'Problem is probably stiff'}

        new_message = None
        try:
            new_message = 'Dopri5 failed with flag {0}: {1}'.format(flag, FLAG_DOCUMENTATION[flag])
            exception = Exception(new_message)
        except KeyError:
            # We have no documentation for this exception, let's just reraise it
            exception = solver_exception

        # Use the superclass method to rethrow the exception with our wrapper
        super(Dopri5Solver, self)._handle_solver_exception(exception)

class LSODARSolver(SolverBase, UniqueNameInitialisationMixin):

    @property
    def _solver_exception_class(self):
        from assimulo.exception import ODEPACK_Exception
        return ODEPACK_Exception

    def _default_solver_instance(self):
        from assimulo.solvers import LSODAR

        return LSODAR(self._assimulo_problem)

    @classmethod
    def unique_name(self):
        return 'lsodar'

    def _handle_solver_exception(self, solver_exception):
        flag = parse_flag(solver_exception.message)

        from assimulo.exception import ODEPACK_Exception

        FLAG_DOCUMENTATION = {-1: 'Excess work done on this call (perhaps wrong jt)',
                              -2: 'Excess accuracy requested (tolerances too small)',
                              -3: 'Illegal input detected (see printed message)',
                              -4: 'Repeated error test failures (check all inputs)',
                              -5: 'Repeated convergence failures (perhaps bad jacobian supplied or wrong choice of '
                                  'jt or tolerances)',
                              -6: 'Error weight became zero during problem.',
                              -7: 'Work space insufficient to finish (see messages)'}
        new_message = None
        try:
            new_message = 'LSODAR failed with flag {0}: {1}'.format(flag, FLAG_DOCUMENTATION[flag])
            exception = ODEPACK_Exception(new_message)
        except KeyError:
            # We have no documentation for this exception, let's just reraise it
            exception = solver_exception

        # Use the superclass method to rethrow the exception with our wrapper
        super(LSODARSolver, self)._handle_solver_exception(exception)

class ExplicitEulerSolver(SolverBase, UniqueNameInitialisationMixin):

    def _default_solver_instance(self):
        from assimulo.solvers import ExplicitEuler
        return ExplicitEuler(self._assimulo_problem)

    @classmethod
    def unique_name(cls):
        return 'euler'

    def simulate(self, timepoints):
        # Euler solver does not return the correct timepoints for some reason, work around that by resampling them
        trajectories = super(ExplicitEulerSolver, self).simulate(timepoints)

        resampled_trajectories = []
        for trajectory in trajectories:
            resampled_trajectories.append(trajectory.resample(timepoints))

        return resampled_trajectories


class RungeKutta4Solver(SolverBase, UniqueNameInitialisationMixin):

    def _default_solver_instance(self):
        from assimulo.solvers import RungeKutta4

        return RungeKutta4(self._assimulo_problem)

    @classmethod
    def unique_name(cls):
        return 'rungekutta4'

    def simulate(self, timepoints):
        # RungeKutta4 solver does not return the correct timepoints for some reason, work around that by resampling them
        trajectories = super(RungeKutta4Solver, self).simulate(timepoints)

        resampled_trajectories = []
        for trajectory in trajectories:
            resampled_trajectories.append(trajectory.resample(timepoints))

        return resampled_trajectories

class RungeKutta34Solver(SolverBase, UniqueNameInitialisationMixin):

    def _default_solver_instance(self):
        from assimulo.solvers import RungeKutta34

        return RungeKutta34(self._assimulo_problem)

    @classmethod
    def unique_name(cls):
        return 'rungekutta34'

class Radau5Solver(SolverBase, UniqueNameInitialisationMixin):

    def _default_solver_instance(self):
        from assimulo.solvers import Radau5ODE

        return Radau5ODE(self._assimulo_problem)

    @classmethod
    def unique_name(cls):
        return 'radau5'

    def _handle_solver_exception(self, solver_exception):
        # Let's try and parse the exception flag, to add some helpful info
        flag = parse_flag(solver_exception.message)

        FLAG_DOCUMENTATION = {-1: 'Input is not consistent',
                              -2: 'Larger NMAX is needed',
                              -3: 'Step size becomes too small',
                              -4: 'Matrix is repeatedly singular'}

        new_message = None
        try:
            new_message = 'Radau5 failed with flag {0}: {1}'.format(flag, FLAG_DOCUMENTATION[flag])
            exception = Exception(new_message)
        except KeyError:
            # We have no documentation for this exception, let's just reraise it
            exception = solver_exception

        # Use the superclass method to rethrow the exception with our wrapper
        super(Radau5Solver, self)._handle_solver_exception(exception)

class RodasSolver(SolverBase, UniqueNameInitialisationMixin):

    def _default_solver_instance(self):
        from assimulo.solvers import RodasODE
        return RodasODE(self._assimulo_problem)

    @classmethod
    def unique_name(cls):
        return 'rodas'

    def _handle_solver_exception(self, solver_exception):
        # Let's try and parse the exception flag, to add some helpful info
        flag = parse_flag(solver_exception.message)

        FLAG_DOCUMENTATION = {-1: 'Input is not consistent',
                              -2: 'Larger NMAX is needed',
                              -3: 'Step size becomes too small',
                              -4: 'Matrix is repeatedly singular'}

        new_message = None
        try:
            new_message = 'Rodas failed with flag {0}: {1}'.format(flag, FLAG_DOCUMENTATION[flag])
            exception = Exception(new_message)
        except KeyError:
            # We have no documentation for this exception, let's just reraise it
            exception = solver_exception

        # Use the superclass method to rethrow the exception with our wrapper
        super(RodasSolver, self)._handle_solver_exception(exception)

#-- Solvers with sensitivity support -----------------------------------------------------------------------------------


def _add_sensitivity_data_to_trajectories(trajectories, raw_sensitivity_data, parameters):
    sensitivity_values = []
    for i, trajectory in enumerate(trajectories):
        ode_term = trajectory.description
        term_sensitivities = []
        for j, parameter in enumerate(parameters):
            term_sensitivities.append((parameter, raw_sensitivity_data[j, :, i]))
        sensitivity_values.append(term_sensitivities)
    trajectories_with_sensitivity_data = []
    for trajectory, sensitivities in zip(trajectories, sensitivity_values):

        # Collect the sensitivities into a nice dictionary of Trajectory objects
        sensitivity_trajectories = []
        for parameter, values in sensitivities:
            sensitivity_trajectories.append(Trajectory(trajectory.timepoints, values,
                                                       SensitivityTerm(trajectory.description, parameter)))

        trajectory_with_sensitivities = TrajectoryWithSensitivityData.from_trajectory(trajectory,
                                                                                      sensitivity_trajectories)
        trajectories_with_sensitivity_data.append(trajectory_with_sensitivities)
    return trajectories_with_sensitivity_data


class SensitivitySolverBase(SolverBase):

    @property
    def _assimulo_problem(self):
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


    def _results_to_trajectories(self, simulated_timepoints, simulated_values):
        trajectories = super(SensitivitySolverBase, self)._results_to_trajectories(simulated_timepoints,
                                                                                   simulated_values)
        sensitivities_raw = np.array(self._solver.p_sol)

        trajectories_with_sensitivity_data = _add_sensitivity_data_to_trajectories(trajectories, sensitivities_raw,
                                                                                   self._problem.parameters)

        return trajectories_with_sensitivity_data

class CVodeSolverWithSensitivities(SensitivitySolverBase, CVodeMixin):

    def _default_solver_instance(self):
        solver = self._cvode_instance(self._assimulo_problem, self._options)
        # It is necessary to set usesens to true here as we are non-parametric here
        solver.usesens = True
        solver.report_continuously = True
        return solver

class ODE15sSolverWithSensitivities(SensitivitySolverBase, ODE15sMixin):

     def _default_solver_instance(self):
        solver = self._cvode_instance(self._assimulo_problem, self._options)
        # It is necessary to set usesens to true here as we are non-parametric here
        solver.usesens = True
        solver.report_continuously = True
        return solver
