#################################################################
# Functions used for parameter inference.  Selected rate constants
# and initial conditions are varied to minimise the cost function.
# Minimization uses the Nelder-Mead simplex algorithm (python fmin).
# The default cost function calculates the distance (sum of squared
# differences) between the sample data moments and moments 
# calculated using MFK at each of the specified timepoints.
#################################################################

import numpy as np
from scipy.optimize import fmin
from sympy import Symbol

from means.inference.distances import get_distance_function
from means.inference.hypercube import hypercube
from means.inference.results import InferenceResultsCollection, InferenceResult, SolverErrorConvergenceStatus, \
    NormalConvergenceStatus
from means.util.decorators import memoised_property
from means.approximation.ode_problem import Moment
from means.simulation import Simulation, NP_FLOATING_POINT_PRECISION, Trajectory


__all__ = ['Inference', 'InferenceWithRestarts']

# value returned if parameters, means or variances < 0
FTOL = 0.000001
MAX_DIST = float('inf')

class TooManySolverExceptions(Exception):
    """
    Exception that is raised when we had too many solver exceptions for the particular round of optimisation
    """
    def __init__(self, last_guess, *args, **kwargs):
        super(TooManySolverExceptions, self).__init__(*args, **kwargs)
        self.last_guess = last_guess


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


def parse_experimental_data_file(sample):
    """
    Reads sample data and returns a list of timepoints and the trajectories
    :param sample: file containing the sample data
    :return: `timepoints, trajectories`
    """
    def _parse_data_point(data_point):
        try:
            return NP_FLOATING_POINT_PRECISION(data_point)
        except ValueError:
            if data_point == 'N':
                return np.nan
            else:
                raise

    datafile = open(sample,'r')
    trajectories = []
    timepoints = None
    try:
        for l in datafile:
            l = l.strip()
            if not l or l.startswith('>') or l.startswith('#'):
                continue

            data = l.split('\t')
            header = data.pop(0)
            if header == 'time':
                timepoints = np.array(map(NP_FLOATING_POINT_PRECISION, data), dtype=NP_FLOATING_POINT_PRECISION)
            else:
                data = np.array(map(_parse_data_point, data), dtype=NP_FLOATING_POINT_PRECISION)
                moment_str_list = header.split(',')
                n_vec = map(int, moment_str_list)
                moment = Moment(n_vec)
                assert(timepoints is not None)
                assert(timepoints.shape == data.shape)
                trajectories.append(Trajectory(timepoints, data, moment))
    finally:
        datafile.close()

    return timepoints, trajectories


def _mom_indices(problem, mom_names):
    """
    From list of moments in sample file (`mom_names`), identify the indices of the corresponding moments
    in simulated trajectories produced by CVODE. Returns these indices as a list (`mom_index_list`).

    Also returns a list of moments produced by MFK/CVODE (`moments_list)
    :param problem: ODEProblem
    :type problem: ODEProblem
    :param mom_names: List of moments in sample file
    :return: `mom_index_list, moments_list`
    """

    # Get list of moments from mfkoutput to create labels for output data file
    moments_list = [moment.n_vector for moment in problem.ordered_descriptions]

    # Get indices in CVODE solutions for the moments in sample data
    # TODO: terribly inefficient but to be replaced by Trajectories
    mom_index_list = []
    for mom_name in mom_names:
        for i, moment in enumerate(moments_list):
            if (mom_name == moment).all():
                mom_index_list.append(i)
                break

    return mom_index_list, moments_list



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
    # TODO: it is questionable whether we should hardcode this or create a similar interface as in limits
    if any(j < 0 for j in initial_conditions[0:number_of_species]):
        return True

    return False


class InferenceWithRestarts(object):
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

    __method = None
    __simulation_kwargs = None


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
                 variable_parameters, observed_trajectories, method='sum_of_squares',
                 return_intermediate_solutions=False, number_of_processes=1,
                 **simulation_kwargs):
        """

        :param problem: Problem to infer parameters for
        :type problem: :class:`~means.approximation.ode_problem.ODEProblem`
        :param number_of_samples: Number of the starting points to randomly pick
        :param starting_parameter_ranges: Valid initialisation ranges for the parameters
        :param starting_conditions_ranges: Valid initialisation ranges for the initial conditions.
                                           If some initial conditions are not set, they will default to 0.
        :param variable_parameters: A dictionary of variable parameters, in the format
                                    ``{parameter_symbol: (min_value, max_value)}`` where the range
                                    ``(min_value, max_value)`` is the range of the allowed parameter values.
                                    If the range is None, parameters are assumed to be unbounded.
        :param observed_trajectories: A list of `Trajectory` objects containing observed data values.
        :param method: Method of calculating the data fit. Currently supported values are
              - 'sum_of_squares' -  min sum of squares optimisation
              - 'gamma' - maximum likelihood optimisation assuming gamma distribution
              - 'normal'- maximum likelihood optimisation assuming normal distribution
              - 'lognormal' - maximum likelihood optimisation assuming lognormal distribution
        :param return_intermediate_solutions: Return the intermediate parameter solutions that optimisation
                                              routine considered as well.
        :param simulation_kwargs: Keyword arguments to pass to the :class:`means.simulation.Simulation` instance
        :param number_of_processes: If set to more than 1, the inference routines will be paralellised
                                    using ``multiprocessing`` module
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

        self.__method = method
        self._return_intermediate_solutions = return_intermediate_solutions
        self.__simulation_kwargs = simulation_kwargs
        self._number_of_processes = int(number_of_processes)

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
                                                        method=self.method,
                                                        return_intermediate_solutions=self._return_intermediate_solutions,
                                                        **self.simulation_kwargs))

        return inference_objects

    def infer(self):
        if self._number_of_processes <= 1:
            results = map(lambda x: x.infer(), self._inference_objects)
        else:
            import multiprocessing

            inference_objects = self._inference_objects
            p = multiprocessing.Pool(self._number_of_processes, initializer=_multiprocessing_pool_initialiser,
                                     initargs=[inference_objects])
            results = p.map(_multiprocessing_apply_infer, range(len(inference_objects)))
            p.close()

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
    def method(self):
        return self.__method

    @property
    def simulation_kwargs(self):
        return self.__simulation_kwargs


def _multiprocessing_pool_initialiser(objects):
    global inference_objects  # Global is ok here as this function will be called for each process on separate threads
    inference_objects = objects

def _multiprocessing_apply_infer(object_id):
    """
    Used in the InferenceWithRestarts class.
    Needs to be in global scope for multiprocessing module to pick it up

    """
    global inference_objects
    return inference_objects[object_id]._infer_raw()


class Inference(object):

    __problem = None
    __starting_parameters_with_variability = None
    __starting_conditions_with_variability = None
    __constraints = None
    __observed_timepoints = None
    __observed_trajectories = None
    _method = None
    _simulation = None
    __return_itnermediate_solutions = None

    __simulation_kwargs = None

    def __init__(self, problem, starting_parameters, starting_conditions,
                 variable_parameters, observed_trajectories, method='sum_of_squares',
                 return_intermediate_solutions=False,
                 **simulation_kwargs):
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
        :param method: Method of calculating the data fit. Currently supported values are
              `'sum_of_squares'`
                    minimisation of the sum of squares distance between trajectories
              `'gamma'`
                    maximum likelihood optimisation assuming gamma distribution
              `'normal'`
                    maximum likelihood optimisation assuming normal distribution
              `'lognormal'`
                    maximum likelihood optimisation assuming lognormal distribution
        :param return_intermediate_solutions: Return the intermediate parameter solutions that optimisation
                                              routine considered as well.
        :param simulation_kwargs: Keyword arguments to pass to the :class:`means.simulation.Simulation` instance
        """
        self.__problem = problem

        variable_parameters = self._validate_variable_parameters(problem, variable_parameters)

        assert(len(starting_parameters) == len(problem.constants))

        if len(starting_conditions) < problem.number_of_equations:
            starting_conditions = starting_conditions[:] \
                                  + [0.0] * (problem.number_of_equations - len(starting_conditions))


        starting_parameters_with_variability, parameter_constraints = \
            self._generate_values_with_variability_and_constraints(self.problem.constants, starting_parameters,
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

        self._method = method
        self.__simulation_kwargs = simulation_kwargs

        self._return_intermediate_solutions = return_intermediate_solutions

    @memoised_property
    def _simulation(self):
        return Simulation(self.problem, **self.simulation_kwargs)

    @property
    def simulation_kwargs(self):
        return self.__simulation_kwargs

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
            if parameter not in problem.left_hand_side and parameter not in problem.constants:
                raise KeyError('Unknown variable parameter {0!r} provided. '
                               'It is not in the problem\'s parameter list, nor in the left-hand-side of equations')

        return variable_parameters_symbolic



    @memoised_property
    def _distance_between_trajectories_function(self):
        return get_distance_function(self.method)

    def _infer_raw(self):

        initial_guess = _to_guess(self.starting_parameters_with_variability, self.starting_conditions_with_variability)
        observed_trajectories_lookup = self.observed_trajectories_lookup

        problem = self.problem
        starting_conditions_with_variability = self.starting_conditions_with_variability
        starting_parameters_with_variability = self.starting_parameters_with_variability

        timepoints_to_simulate = self.observed_timepoints

        _distance_between_trajectories_function = self._distance_between_trajectories_function

        self.exception_count = 0
        exception_limit = 1

        def distance(current_guess):
            if not self._constraints_are_satisfied(current_guess):
                return MAX_DIST

            current_parameters, current_initial_conditions = _extract_params_from_i0(current_guess,
                                                                        starting_parameters_with_variability,
                                                                        starting_conditions_with_variability)

            if _some_params_are_negative(problem, current_parameters, current_initial_conditions):
                return MAX_DIST

            simulator = self._simulation
            try:
                simulated_trajectories = simulator.simulate_system(current_parameters,
                                                                   current_initial_conditions,
                                                                   timepoints_to_simulate)
            except Exception as e:
                self.exception_count += 1
                if self.exception_count < exception_limit:
                    print 'Warning: got {0!r} while simulating with '  \
                          'parameters={1!r}, initial_conditions={2!r}. ' \
                          'Setting distance to infinity'.format(e, current_parameters, current_initial_conditions)
                    return MAX_DIST
                else:
                    raise TooManySolverExceptions(current_guess, 'Solver exception limit reached '
                                                                 'while exploring the inference space.')

            dist = _distance_between_trajectories_function(simulated_trajectories, observed_trajectories_lookup)
            return dist

        try:
            result = fmin(distance, initial_guess, ftol=FTOL, disp=0, full_output=True,
                          retall=self._return_intermediate_solutions)
        except TooManySolverExceptions as e:
            print 'Warning: Reached maximum number of exceptions from solver. Stopping inference'
            optimised_data = e.last_guess
            distance_at_minimum = MAX_DIST
            convergence_status = SolverErrorConvergenceStatus()
            all_vecs = None
        except Exception:
            raise
        else:
            if self._return_intermediate_solutions:
                optimised_data, distance_at_minimum, iterations_taken, function_calls_made, warning_flag, all_vecs = result
            else:
                optimised_data, distance_at_minimum, iterations_taken, function_calls_made, warning_flag = result
                all_vecs = None

            convergence_status = NormalConvergenceStatus(warning_flag, iterations_taken, function_calls_made)

        optimal_parameters, optimal_initial_conditions = _extract_params_from_i0(optimised_data,
                                                                                self.starting_parameters_with_variability,
                                                                                self.starting_conditions_with_variability)

        if all_vecs is not None:
            solutions = []

            for v in all_vecs:
                solutions.append(_extract_params_from_i0(v, self.starting_parameters_with_variability,
                                                         self.starting_conditions_with_variability))
        else:
            solutions = None

        return optimal_parameters, optimal_initial_conditions, distance_at_minimum, convergence_status, solutions

    def _result_from_raw_result(self, raw_result):
        optimal_parameters, optimal_initial_conditions, \
               distance_at_minimum, convergence_status, solutions = raw_result

        result = InferenceResult(self.problem, self.observed_trajectories,
                                 self.starting_parameters, self.starting_conditions,
                                 optimal_parameters, optimal_initial_conditions,
                                 distance_at_minimum,
                                 convergence_status,
                                 solutions,
                                 self._simulation)
        return result


    def infer(self):
        raw_result = self._infer_raw()
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

    def _constraints_are_satisfied(self, current_guess):
        return _constraints_are_satisfied(current_guess, self.constraints)


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
    def method(self):
        return self._method