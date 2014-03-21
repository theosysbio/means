"""
Parameter Inference
-----

This part of the package provides utilities for parameter inference.
Parameter inference will try to find the set of parameters which
produces trajectories with minimal distance to the observed trajectories.
Different distance functions are implemented (e.g. based on parametric likelihood),
but it is also possible to use custom distance functions.
Parameter inference support parallel computing and can use random
restarts (see :class:`~means.inference.InferenceWithRestarts`).
"""

from scipy.optimize import fmin
from sympy import Symbol

from means.inference.distances import get_distance_function
from means.inference.hypercube import hypercube
from means.inference.parallelisation import raw_results_in_parallel
from means.inference.results import InferenceResultsCollection, InferenceResult, SolverErrorConvergenceStatus, \
    NormalConvergenceStatus
from means.io.serialise import SerialisableObject
from means.simulation import SolverException, Simulation
from means.util.logs import get_logger
from means.util.memoisation import memoised_property, MemoisableObject

logger = get_logger(__name__)

DEFAULT_SOLVER_EXCEPTIONS_LIMIT = 100

__all__ = ['Inference', 'InferenceWithRestarts']

# value returned if parameters, means or variances < 0
FTOL = 0.000001
MAX_DIST = float('inf')

class TooManySolverExceptions(Exception):
    """
    Exception that is raised when we had too many solver exceptions for the particular round of optimisation
    """
    def __init__(self, *args, **kwargs):
        super(TooManySolverExceptions, self).__init__(*args, **kwargs)


def _to_guess(parameters_with_variability, initial_conditions_with_variability):
    """
    Creates a list of variables to infer, based on the values in vary/varyic (0=fixed, 1=optimised).

    This should contain all variables that are varied as it would be passed to optimisation method.

    :param param: list of starting values for kinetic parameters
    :param initcond: list of starting values (i.e. at t0) for moments
    :return: i0 (which is passed to the fmin minimisation function)
    """
    # Return all the items that have variable=True
    return [x[0] for x in parameters_with_variability + initial_conditions_with_variability if x[1]]



def _extract_params_from_i0(only_variable_parameters, parameters_with_variability, initial_conditions_with_variability):
    """
    Used within the distance/cost function to create the current kinetic parameter and initial condition vectors
    to be used during that interaction, using current values in i0.

    This function takes i0 and complements it with additional information from variables that we do not want to vary
    so the simulation function could be run and values compared.

    :param only_variable_parameters: `i0` list returned from `make_i0`
    :param param: list of starting values for kinetic parameters
    :param vary: list to identify which values in `param` to vary during inference (0=fixed, 1=optimise)
    :param initcond: list of starting values (i.e. at t0) for moments
    :param varyic: list to identify which values in `initcond` to vary (0=fixed, 1=optimise)
    :return:
    """

    complete_params = []
    counter = 0
    for param, is_variable in parameters_with_variability:
        # If param not variable, add it from param list
        if not is_variable:
            complete_params.append(param)
        else:
            # Otherwise add it from variable parameters list
            complete_params.append(only_variable_parameters[counter])
            counter += 1

    complete_initial_conditions = []
    for initial_condition, is_variable in initial_conditions_with_variability:
        if not is_variable:
            complete_initial_conditions.append(initial_condition)
        else:
            complete_initial_conditions.append(only_variable_parameters[counter])
            counter += 1

    return complete_params, complete_initial_conditions

def _constraints_are_satisfied(current_guess, limits):
    if limits is not None:
        for value, limit in zip(current_guess, limits):
            if limit is None:
                continue

            lower_limit = limit[0]
            upper_limit = limit[1]
            if lower_limit:
                if value < lower_limit:
                    return False
            if upper_limit:
                if value > upper_limit:
                    return False

    return True

def _some_params_are_negative(problem, parameters, initial_conditions):
    number_of_species = problem.number_of_species
    if any(i < 0 for i in parameters):     # parameters cannot be negative
        return True
    # disallow negative numbers for raw moments (i.e. cannot have -1.5 molecules on avg)
    if any(j < 0 for j in initial_conditions[0:number_of_species]):
        return True

    return False


class InferenceWithRestarts(MemoisableObject):
    """
    Parameter Inference Method that utilises multiple seed points for the optimisation.

    """

    __problem = None
    __number_of_samples = None
    __starting_parameter_ranges = None
    __starting_conditions_ranges = None
    __variable_parameters = None
    __observed_trajectories = None
    _return_intermediate_solutions = None

    __distance_function_type = None


    def _validate_range(self, range_):
        validated_range = []
        for item in range_:
            try:
                item = tuple(map(float, item))
            except (ValueError, TypeError):
                raise ValueError('Invalid range provided: expected ``(min_value, max_value)``, '
                                 'where min_value and max_value are both floats, got {0!r} instead'.format(item))
            if len(item) != 2 or item[0] > item[1]:
                raise ValueError('Invalid range provided: expected: ``(min_value, max_value)`` '
                                 'where ``min_value < max_value``, got {0!r} instead'.format(item))

            validated_range.append(item)

        return validated_range


    def __init__(self, problem, number_of_samples,
                 starting_parameter_ranges, starting_conditions_ranges,
                 variable_parameters, observed_trajectories, distance_function_type='sum_of_squares'):
        """

        :param problem: Problem to infer parameters for
        :type problem: :class:`~means.core.ODEProblem`
        :param number_of_samples: Number of the starting points to randomly pick
        :param starting_parameter_ranges: Valid initialisation ranges for the parameters
        :param starting_conditions_ranges: Valid initialisation ranges for the initial conditions.
                                           If some initial conditions are not set, they will default to 0.
        :param variable_parameters: A dictionary of variable parameters, in the format
                                    ``{parameter_symbol: (min_value, max_value)}`` where the range
                                    ``(min_value, max_value)`` is the range of the allowed parameter values.
                                    If the range is None, parameters are assumed to be unbounded.
        :param observed_trajectories: A list of `Trajectory` objects containing observed data values.
        :param distance_function_type: Method of calculating the data fit. Currently supported values are
              - 'sum_of_squares' -  min sum of squares optimisation
              - 'gamma' - maximum likelihood optimisation assuming gamma distribution
              - 'normal'- maximum likelihood optimisation assuming normal distribution
              - 'lognormal' - maximum likelihood optimisation assuming lognormal distribution
              - any callable function, that takes two arguments: simulated trajectories (list)
                and observed trajectories lookup (dictionary of description: trajectory pairs)
                see :func:`means.inference.distances.sum_of_squares` for examples of such functions
        """

        self.__problem = problem
        self.__number_of_samples = number_of_samples


        starting_parameter_ranges = self._validate_range(starting_parameter_ranges)
        self.__starting_parameter_ranges = starting_parameter_ranges

        if len(starting_parameter_ranges) != problem.number_of_parameters:
            raise ValueError('Incorrect number of parameter ranges provided. '
                             'Expected exactly {0}, got {1}'.format(problem.number_of_parameters,
                                                            len(starting_parameter_ranges)))

        starting_conditions_ranges = self._validate_range(starting_conditions_ranges)

        if len(starting_conditions_ranges) > problem.number_of_equations:
            raise ValueError('Incorrect number of parameter ranges provided. '
                             'Expected at most {0}, got {1}'.format(problem.number_of_equations,
                                                                    len(starting_conditions_ranges)))

        self.__starting_conditions_ranges = starting_conditions_ranges

        self.__variable_parameters = variable_parameters
        self.__observed_trajectories = observed_trajectories
        if not observed_trajectories:
            raise ValueError('No observed trajectories provided. Need at least one to perform parameter inference')

        self.__distance_function_type = distance_function_type

    @memoised_property
    def _inference_objects(self):

        full_list_of_ranges = self.starting_parameter_ranges[:] + self.starting_conditions_ranges[:]
        variables_collection = hypercube(self.number_of_samples, full_list_of_ranges)

        inference_objects = []
        for variables in variables_collection:
            starting_parameters = variables[:len(self.starting_parameter_ranges)]
            starting_conditions = variables[len(self.starting_parameter_ranges):]

            inference_objects.append(Inference(self.problem,
                                                starting_parameters,
                                                starting_conditions,
                                                self.variable_parameters,
                                                self.observed_trajectories,
                                                distance_function_type=self.distance_function_type,
                                                ))

        return inference_objects

    def infer(self, number_of_processes=1, *args, **kwargs):
        """
        :param number_of_processes: If set to more than 1, the inference routines will be paralellised
                                    using ``multiprocessing`` module
        :param args: arguments to pass to :meth:`Inference.infer`
        :param kwargs: keyword arguments to pass to :meth:`Inference.infer`
        :return:
        """
        if number_of_processes == 1:
            results = map(lambda x: x.infer(*args, **kwargs), self._inference_objects)
        else:
            inference_objects = self._inference_objects
            results = raw_results_in_parallel(self._inference_objects, number_of_processes, *args,
                                              **kwargs)
            results = [inference._result_from_raw_result(raw_result)
                       for inference, raw_result in zip(inference_objects, results)]


        results = sorted(results, key=lambda x: x.distance_at_minimum)

        return InferenceResultsCollection(results)

    @property
    def problem(self):
        """
        :rtype: ODEProblem
        """
        return self.__problem

    @property
    def number_of_samples(self):
        return self.__number_of_samples

    @property
    def starting_parameter_ranges(self):
        return self.__starting_parameter_ranges

    @property
    def starting_conditions_ranges(self):
        return self.__starting_conditions_ranges

    @property
    def variable_parameters(self):
        return self.__variable_parameters

    @property
    def observed_trajectories(self):
        return self.__observed_trajectories

    @property
    def distance_function_type(self):
        return self.__distance_function_type

class Inference(SerialisableObject, MemoisableObject):

    __problem = None
    __starting_parameters_with_variability = None
    __starting_conditions_with_variability = None
    __constraints = None
    __observed_timepoints = None
    __observed_trajectories = None
    _distance_function_type = None
    _variable_parameters = None

    yaml_tag = '!inference'

    @classmethod
    def to_yaml(cls, dumper, data):
        # Variable parameters are assumed to be validated here and only in {symbol : range_} format
        variable_parameters = data.variable_parameters
        # Convert key to string as sympy is a bit too smart and does not allow sorting symbols
        variable_parameters = {str(key): value for key, value in variable_parameters.iteritems()}

        mapping = [('problem', data.problem),
                   ('starting_parameters', data.starting_parameters),
                   ('starting_conditions', data.starting_conditions),
                   ('variable_parameters', variable_parameters),
                   ('observed_trajectories', data.observed_trajectories),
                   ('distance_function_type', data.distance_function_type)]

        mapping.extend(data.simulation_kwargs.items())

        return dumper.represent_mapping(cls.yaml_tag, mapping)

    def __init__(self, problem, starting_parameters, starting_conditions,
                 variable_parameters, observed_trajectories, distance_function_type='sum_of_squares', **simulation_kwargs):
        """

        :param problem: ODEProblem to infer data for
        :type problem: ODEProblem
        :param starting_parameters: A list of starting values for each of the model's parameters
        :type starting_parameters: iterable
        :param starting_conditions: A list of starting values for each of the initial conditions.
                                    All unspecified initial conditions will be set to zero
        :type starting_conditions: iterable
        :param variable_parameters: A dictionary of variable parameters, in the format
                                    ``{parameter_symbol: (min_value, max_value)}`` where the range
                                    ``(min_value, max_value)`` is the range of the allowed parameter values.
                                    If the range is None, parameters are assumed to be unbounded.
        :param observed_trajectories: A list of `Trajectory` objects containing observed data values.
        :param distance_function_type: Method of calculating the data fit. Currently supported values are
              `'sum_of_squares'`
                    minimisation of the sum of squares distance between trajectories
              `'gamma'`
                    maximum likelihood optimisation assuming gamma distribution
              `'normal'`
                    maximum likelihood optimisation assuming normal distribution
              `'lognormal'`
                    maximum likelihood optimisation assuming lognormal distribution
        :param simulation_kwargs: Keyword arguments to pass to the :class:`means.simulation.Simulation` instance
        """
        self.__problem = problem

        variable_parameters = self._validate_variable_parameters(problem, variable_parameters)
        self._variable_parameters = variable_parameters

        assert(len(starting_parameters) == len(problem.parameters))

        if len(starting_conditions) < problem.number_of_equations:
            starting_conditions = starting_conditions[:] \
                                  + [0.0] * (problem.number_of_equations - len(starting_conditions))

        starting_parameters_with_variability, parameter_constraints = \
            self._generate_values_with_variability_and_constraints(self.problem.parameters, starting_parameters,
                                                                   variable_parameters)

        starting_conditions_with_variability, initial_condition_constraints = \
            self._generate_values_with_variability_and_constraints(self.problem.left_hand_side, starting_conditions,
                                                                   variable_parameters)

        self.__starting_parameters_with_variability = starting_parameters_with_variability
        self.__starting_conditions_with_variability = starting_conditions_with_variability

        constraints = parameter_constraints + initial_condition_constraints

        assert(constraints is None or len(constraints) == len(filter(lambda x: x[1],
                                                                     starting_parameters_with_variability +
                                                                     starting_conditions_with_variability)))
        self.__constraints = constraints

        self.__observed_trajectories = observed_trajectories
        if not observed_trajectories:
            raise ValueError('No observed trajectories provided. Need at least one to perform parameter inference')

        self.__observed_timepoints = observed_trajectories[0].timepoints

        self._distance_function_type = distance_function_type
        self._simulation_kwargs = simulation_kwargs


    def _generate_values_with_variability_and_constraints(self, symbols, starting_values, variable_parameters):
        """
        Generates the `values_with_variability` formatted list
        from the provided symbols, starting values and variable parameters

        :param symbols: The symbols defining each of the values in the starting values list
        :param starting_values: the actual starting values
        :param variable_parameters: a dictionary/set/list of variables that are variable
                                    if dictionary provided, the contents should be `symbol: range` where range is
                                    a tuple ``(min_val, max_val)`` of allowed parameter values or ``None`` for no limit.
                                    if set/list provided, the ranges will be assumed to be ``None`` for each of
                                    the parameters
        :type variable_parameters: dict|iterable
        :return:
        """
        values_with_variability = []
        constraints = []

        if not isinstance(variable_parameters, dict):
            # Convert non/dict representations to Dict with nones
            variable_parameters = {p: None for p in variable_parameters}

        for parameter, parameter_value in zip(symbols, starting_values):
            try:
                constraint = variable_parameters[parameter]
                variable = True
            except KeyError:
                try:
                    constraint = variable_parameters[str(parameter)]
                    variable = True
                except KeyError:
                    constraint = None
                    variable = False

            values_with_variability.append((parameter_value, variable))
            if variable:
                constraints.append(constraint)

        return values_with_variability, constraints

    def _validate_variable_parameters(self, problem, variable_parameters):
        if not variable_parameters:
            raise ValueError("No variable parameters specified, nothing to infer")

        if not isinstance(variable_parameters, dict):
            variable_parameters = {p: None for p in variable_parameters}

        variable_parameters_symbolic = {}
        for parameter, range_ in variable_parameters.iteritems():
            if not isinstance(parameter, Symbol):
                parameter = Symbol(parameter)
            if range_ is not None:
                try:
                    range_ = tuple(map(float, range_))
                except (TypeError, ValueError):
                    raise ValueError('Invalid range provided for {0!r} - '
                                     'expected tuple of floats, got {1!r}'.format(parameter, range_))

                if len(range_) != 2:
                    raise ValueError('Invalid range provided for {0!r} - '
                                     'expected tuple of length two, got: {1!r}'.format(parameter, range_))

            variable_parameters_symbolic[parameter] = range_

        for parameter in variable_parameters_symbolic:
            if parameter not in problem.left_hand_side and parameter not in problem.parameters:
                raise KeyError('Unknown variable parameter {0!r} provided. '
                               'It is not in the problem\'s parameter list, nor in the left-hand-side of equations')

        return variable_parameters_symbolic



    @memoised_property
    def _distance_between_trajectories_function(self):
        return get_distance_function(self.distance_function_type)

    class _DistancesCalculator(object):

        def __init__(self, problem, constraints,
                     parameters_with_variability, initial_conditions_with_variability,
                     timepoints_to_simulate, observed_trajectories_lookup,
                     distance_comparison_function,
                     simulation_instance,
                     exception_limit):

            self.problem = problem
            self.constraints = constraints

            self.parameters_with_variability = parameters_with_variability
            self.initial_conditions_with_variability = initial_conditions_with_variability
            self.simulation_instance = simulation_instance
            self.distance_comparison_function = distance_comparison_function
            self.timepoints_to_simulate = timepoints_to_simulate
            self.observed_trajectories_lookup = observed_trajectories_lookup
            self.exception_limit = exception_limit
            self.exception_count = 0
            self.best_so_far_distance = None
            self.best_so_far_guess = None


        def _constraints_are_satisfied(self, current_guess):
            return _constraints_are_satisfied(current_guess, self.constraints)

        def extract_parameters_from_optimisation_guess(self, current_guess):
            return _extract_params_from_i0(current_guess,
                                           self.parameters_with_variability, self.initial_conditions_with_variability)

        def _distance_to_simulated_trajectories(self, simulated_trajectories):
            return self.distance_comparison_function(simulated_trajectories, self.observed_trajectories_lookup)

        def __call__(self, current_guess):
            if not self._constraints_are_satisfied(current_guess):
                return MAX_DIST

            current_parameters, current_initial_conditions = self.extract_parameters_from_optimisation_guess(current_guess)

            if _some_params_are_negative(self.problem, current_parameters, current_initial_conditions):
                return MAX_DIST

            simulator = self.simulation_instance
            try:
                simulated_trajectories = simulator.simulate_system(current_parameters,
                                                                   current_initial_conditions,
                                                                   self.timepoints_to_simulate)
            except SolverException as e:
                logger.warn('Warning: got {0!r} while simulating with '  \
                             'parameters={1!r}, initial_conditions={2!r}. ' \
                             'Setting distance to infinity'.format(e, current_parameters, current_initial_conditions))
                self.exception_count += 1
                if self.exception_limit is not None and self.exception_count > self.exception_limit:
                    raise TooManySolverExceptions('Solver exception limit reached while exploring the inference space.')
                else:
                    return MAX_DIST


            dist = self._distance_to_simulated_trajectories(simulated_trajectories)
            # Keep track of the best-so-far score if we cancel early due to too many exceptions
            if dist < self.best_so_far_distance:
                self.best_so_far_distance = dist
                self.best_so_far_guess = current_guess

            return dist


    def _infer_raw(self, return_intermediate_solutions=False, solver_exceptions_limit=DEFAULT_SOLVER_EXCEPTIONS_LIMIT):

        initial_guess = _to_guess(self.starting_parameters_with_variability, self.starting_conditions_with_variability)

        distances_calculator = self._DistancesCalculator(self.problem,
                                                         self.constraints,
                                                         self.starting_parameters_with_variability,
                                                         self.starting_conditions_with_variability,
                                                         self.observed_timepoints,
                                                         self.observed_trajectories_lookup,
                                                         self._distance_between_trajectories_function,
                                                         self.simulation,
                                                         exception_limit=solver_exceptions_limit)

        try:
            result = fmin(distances_calculator, initial_guess, ftol=FTOL, disp=0, full_output=True,
                          retall=return_intermediate_solutions)
        except TooManySolverExceptions as e:
            logger.warn('Reached maximum number of exceptions from solver. Stopping inference here')
            if distances_calculator.best_so_far_guess is not None:
                optimised_data = distances_calculator.best_so_far_guess
            else:
                optimised_data = initial_guess

            distance_at_minimum = MAX_DIST
            convergence_status = SolverErrorConvergenceStatus()
            all_vecs = None
        except Exception:
            raise
        else:
            if return_intermediate_solutions:
                optimised_data, distance_at_minimum, iterations_taken, function_calls_made, warning_flag, all_vecs = result
            else:
                optimised_data, distance_at_minimum, iterations_taken, function_calls_made, warning_flag = result
                all_vecs = None

            convergence_status = NormalConvergenceStatus(warning_flag, iterations_taken, function_calls_made)

        optimal_parameters, optimal_initial_conditions \
            = distances_calculator.extract_parameters_from_optimisation_guess(optimised_data)

        if all_vecs is not None:
            solutions = []

            for v in all_vecs:
                solutions.append(_extract_params_from_i0(v, self.starting_parameters_with_variability,
                                                         self.starting_conditions_with_variability))
        else:
            solutions = None

        return optimal_parameters, optimal_initial_conditions, distance_at_minimum, convergence_status, solutions

    def _result_from_raw_result(self, raw_result):
        optimal_parameters, optimal_initial_conditions, distance_at_minimum, convergence_status, solutions = raw_result

        result = InferenceResult(self, optimal_parameters, optimal_initial_conditions,
                                 distance_at_minimum, convergence_status,
                                 solutions)
        return result


    def infer(self, return_intermediate_solutions=False, solver_exceptions_limit=DEFAULT_SOLVER_EXCEPTIONS_LIMIT):
        """

        :param return_intermediate_solutions: Return the intermediate parameter solutions that optimisation
        """
        raw_result = self._infer_raw(return_intermediate_solutions,
                                     solver_exceptions_limit=solver_exceptions_limit)
        return self._result_from_raw_result(raw_result)

    @property
    def problem(self):
        """
        :rtype: ODEProblem
        """
        return self.__problem

    @property
    def starting_parameters_with_variability(self):
        return self.__starting_parameters_with_variability

    @property
    def starting_parameters(self):
        return [x[0] for x in self.starting_parameters_with_variability]

    @property
    def starting_conditions_with_variability(self):
        return self.__starting_conditions_with_variability

    @property
    def starting_conditions(self):
        return [x[0] for x in self.starting_conditions_with_variability]

    @property
    def constraints(self):
        return self.__constraints

    @property
    def simulation_kwargs(self):
        return self._simulation_kwargs.copy()

    @memoised_property
    def simulation(self):
        return Simulation(self.problem, **self.simulation_kwargs)

    @property
    def observed_timepoints(self):
        return self.__observed_timepoints

    @property
    def observed_trajectories(self):
        return self.__observed_trajectories

    @memoised_property
    def observed_trajectories_lookup(self):
        """
        Similar to observed_trajectories, but returns a dictionary of {description:trajectory}
        """
        return {trajectory.description: trajectory for trajectory in self.observed_trajectories}

    @property
    def variable_parameters(self):
        return self._variable_parameters

    @property
    def distance_function_type(self):
        return self._distance_function_type


    def __eq__(self, other):
        if not isinstance(other, self.__class__):
             return False

        return self.problem == other.problem \
               and self.starting_conditions == other.starting_conditions \
               and self.starting_parameters == other.starting_parameters \
               and self.variable_parameters == other.variable_parameters \
               and self.observed_trajectories == other.observed_trajectories \
               and self.distance_function_type == other.distance_function_type \
               and self.simulation_kwargs == other.simulation_kwargs