#################################################################
# Functions used for parameter inference.  Selected rate constants
# and initial conditions are varied to minimise the cost function.
# Minimization uses the Nelder-Mead simplex algorithm (python fmin).
# The default cost function calculates the distance (sum of squared
# differences) between the sample data moments and moments 
# calculated using MFK at each of the specified timepoints.
#################################################################
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fmin
from sympy import Symbol
from means.io.serialise import SerialisableObject

from means.util.decorators import memoised_property
from means.inference.gamma_infer import _distribution_distance, SUPPORTED_DISTRIBUTIONS
from means.approximation.ode_problem import Moment
from means.simulation import Simulation, NP_FLOATING_POINT_PRECISION, Trajectory

# value returned if parameters, means or variances < 0
FTOL = 0.000001
MAX_DIST = float('inf')

def to_guess(parameters_with_variability, initial_conditions_with_variability):
    """
    Creates a list of variables to infer, based on the values in vary/varyic (0=fixed, 1=optimised).

    This should contain all variables that are varied as it would be passed to optimisation method.

    :param param: list of starting values for kinetic parameters
    :param initcond: list of starting values (i.e. at t0) for moments
    :return: i0 (which is passed to the fmin minimisation function)
    """
    # Return all the items that have variable=True
    return [x[0] for x in parameters_with_variability + initial_conditions_with_variability if x[1]]



def extract_params_from_i0(only_variable_parameters, parameters_with_variability, initial_conditions_with_variability):
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

def sum_of_squares_distance(simulated_trajectories, observed_trajectories_lookup):
    dist = 0
    for simulated_trajectory in simulated_trajectories:
        observed_trajectory = None
        try:
            observed_trajectory = observed_trajectories_lookup[simulated_trajectory.description]
        except KeyError:
            continue

        deviations = observed_trajectory.values - simulated_trajectory.values
        # Drop NaNs arising from missing datapoints
        deviations = deviations[~np.isnan(deviations)]

        dist += np.sum(np.square(deviations))

    return dist

def constraints_are_satisfied(current_guess, limits):
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

def some_params_are_negative(problem, parameters, initial_conditions):
    number_of_species = problem.number_of_species
    if any(i < 0 for i in parameters):     # parameters cannot be negative
        return True
    # disallow negative numbers for raw moments (i.e. cannot have -1.5 molecules on avg)
    # TODO: it is questionable whether we should hardcode this or create a similar interface as in limits
    if any(j < 0 for j in initial_conditions[0:number_of_species]):
        return True

    return False

class InferenceResult(SerialisableObject):

    __problem = None
    __observed_trajectories = None
    __starting_parameters = None
    __starting_initial_conditions = None

    __optimal_parameters = None
    __optimal_initial_conditions = None

    __distance_at_minimum = None
    __iterations_taken = None
    __function_calls_made = None
    __warning_flag = None
    __solutions = None

    _simulation = None

    yaml_tag = '!inference-result'

    def __init__(self, problem, observed_trajectories, starting_parameters, starting_initial_conditions,
                 optimal_parameters, optimal_initial_conditions, distance_at_minimum, iterations_taken,
                 function_calls_made, warning_flag, solutions, simulation):

        """

        :param problem:
        :param observed_trajectories:
        :param starting_parameters:
        :param starting_initial_conditions:
        :param optimal_parameters:
        :param optimal_initial_conditions:
        :param distance_at_minimum:
        :param iterations_taken:
        :param function_calls_made:
        :param warning_flag:
        :param solutions:
        :param simulation:
        :type simulation: :class:`means.simulation.Simulation`
        """
        self.__problem = problem

        self.__observed_trajectories = observed_trajectories
        self.__starting_parameters = starting_parameters
        self.__starting_initial_conditions = starting_initial_conditions
        self.__optimal_parameters = optimal_parameters
        self.__optimal_initial_conditions = optimal_initial_conditions
        self.__distance_at_minimum = distance_at_minimum
        self.__iterations_taken = iterations_taken
        self.__warning_flag = warning_flag
        self.__function_calls_made = function_calls_made
        self.__solutions = solutions
        self._simulation = simulation

    @property
    def problem(self):
        return self.__problem

    @property
    def observed_trajectories(self):
        return self.__observed_trajectories

    @property
    def starting_parameters(self):
        return self.__starting_parameters

    @property
    def starting_initial_conditions(self):
        return self.__starting_initial_conditions

    @property
    def optimal_parameters(self):
        return self.__optimal_parameters



    @property
    def optimal_initial_conditions(self):
        return self.__optimal_initial_conditions

    @property
    def distance_at_minimum(self):
        return self.__distance_at_minimum

    @property
    def iterations_taken(self):
        return self.__iterations_taken

    @property
    def function_calls_made(self):
        return self.__function_calls_made

    @property
    def warning_flag(self):
        return self.__warning_flag

    @property
    def solutions(self):
        """
        Solutions at each each iteration of optimisation.
        :return: a list of (parameters, conditions) pairs
        :rtype: list[tuple]
        """
        return self.__solutions

    @memoised_property
    def starting_trajectories(self):
        timepoints = self.observed_trajectories[0].timepoints

        starting_trajectories = self._simulation.simulate_system(self.starting_parameters,
                                                                 self.starting_initial_conditions, timepoints)
        return starting_trajectories

    @memoised_property
    def optimal_trajectories(self):
        timepoints = self.observed_trajectories[0].timepoints
        optimal_trajectories = self._simulation.simulate_system(self.optimal_parameters,
                                                                self.optimal_initial_conditions, timepoints)

        return optimal_trajectories

    @memoised_property
    def intermediate_trajectories(self):
        timepoints = self.observed_trajectories[0].timepoints
        return map(lambda x: self._simulation.simulate_system(x[0], x[1], timepoints), self.solutions)

    def plot(self, plot_intermediate_solutions=False):
        """
        Plot the inference result.

        :param plot_intermediate_solutions: plot the trajectories resulting from the intermediate solutions as well
        """
        from matplotlib import pyplot as plt
        observed_trajectories = self.observed_trajectories

        starting_trajectories = self.starting_trajectories
        optimal_trajectories = self.optimal_trajectories

        if plot_intermediate_solutions:
            intermediate_trajectories = self.intermediate_trajectories
        else:
            intermediate_trajectories = []

        for observed_trajectory in observed_trajectories:
            plt.figure()
            description = observed_trajectory.description
            plt.title(description)
            observed_trajectory.plot('+', label="Observed data")
            for trajectory in starting_trajectories:
                if trajectory.description == description:
                    trajectory.plot(label="Starting trajectory")
                    break

            for trajectory in optimal_trajectories:
                if trajectory.description == description:
                    trajectory.plot(label="Optimal trajectory")
                    break

            for i, intermediate_trajectories in enumerate(intermediate_trajectories):
                for trajectory in intermediate_trajectories:
                    if trajectory.description == description:
                        # Only add the label once
                        label = 'Intermediate trajectories' if i == 0 else ''
                        trajectory.plot(label=label, alpha=0.1, color='red')


            plt.legend()



    def __unicode__(self):
        return u"""
        {self.__class__!r}
        Starting Parameters: {self.starting_parameters!r}
        Optimal Parameters: {self.optimal_parameters!r}

        Starting Initial Conditions: {self.starting_initial_conditions!r}
        Optimal Initial Conditions: {self.optimal_initial_conditions!r}

        Distance at Minimum: {self.distance_at_minimum!r}
        Iterations taken: {self.iterations_taken!r}
        """.format(self=self)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return unicode(self).encode('utf8')

    @classmethod
    def to_yaml(cls, dumper, data):
        solutions = []
        for a, b in data.solutions:
            solutions.append((list(a), list(b)))

        mapping = [('problem', data.problem),
                   ('observed_trajectories', data.observed_trajectories),
                   ('starting_parameters', data.starting_parameters),
                   ('starting_initial_conditions', data.starting_initial_conditions),
                   ('optimal_parameters', data.optimal_parameters),
                   ('optimal_initial_conditions', data.optimal_initial_conditions),
                   ('distance_at_minimum', data.distance_at_minimum),
                   ('iterations_taken', data.iterations_taken),
                   ('function_calls_made', data.function_calls_made),
                   ('warning_flag', data.warning_flag),
                   ('solutions', solutions),
                   ('simulation', data._simulation)]

        return dumper.represent_mapping(cls.yaml_tag, mapping)

class ParameterInference(object):

    __problem = None
    __starting_parameters_with_variability = None
    __starting_conditions_with_variability = None
    __constraints = None
    __observed_timepoints = None
    __observed_trajectories = None
    _method = None
    _simulation = None

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


    def __init__(self, problem, starting_parameters, starting_conditions,
                 variable_parameters, observed_timepoints, observed_trajectories, method='sum_of_squares',
                 **simulation_kwargs):
        """

        :param problem: ODEProblem to infer data for
        :type problem: ODEProblem
        :param starting_parameters: A list of starting values for each of the model's parameters
        :type starting_parameters: iterable
        :param starting_conditions: A list of starting vlaues for each of the initial conditions.
                                    All unspecified initial conditions will be set to zero
        :type starting_conditions: iterable
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

        self.__observed_timepoints = observed_timepoints
        self.__observed_trajectories = observed_trajectories

        self._method = method
        self._simulation = Simulation(self.problem, **simulation_kwargs)

    @memoised_property
    def _distance_between_trajectories_function(self):
        if self.method == 'sum_of_squares':
            return sum_of_squares_distance
        elif self.method in SUPPORTED_DISTRIBUTIONS:
            return lambda x, y: _distribution_distance(x, y, self.method)
        else:
            raise ValueError('Unsupported method {0!r}'.format(self.method))

    def infer(self):

        initial_guess = to_guess(self.starting_parameters_with_variability, self.starting_conditions_with_variability)
        observed_trajectories_lookup = self.observed_trajectories_lookup

        problem = self.problem
        starting_conditions_with_variability = self.starting_conditions_with_variability
        starting_parameters_with_variability = self.starting_parameters_with_variability

        timepoints_to_simulate = self.observed_timepoints

        _distance_between_trajectories_function = self._distance_between_trajectories_function
        def distance(current_guess):
            if not self._constraints_are_satisfied(current_guess):
                return MAX_DIST

            current_parameters, current_initial_conditions = extract_params_from_i0(current_guess,
                                                                        starting_parameters_with_variability,
                                                                        starting_conditions_with_variability)

            if some_params_are_negative(problem, current_parameters, current_initial_conditions):
                return MAX_DIST

            simulator = self._simulation
            simulated_trajectories = simulator.simulate_system(current_parameters,
                                                               current_initial_conditions,
                                                               timepoints_to_simulate)

            dist = _distance_between_trajectories_function(simulated_trajectories, observed_trajectories_lookup)
            return dist

        optimised_data, distance_at_minimum, iterations_taken, function_calls_made, warning_flag, all_vecs \
            = fmin(distance, initial_guess, ftol=FTOL, disp=0, full_output=True, retall=True)

        optimal_parameters, optimal_initial_conditions = extract_params_from_i0(optimised_data,
                                                                                self.starting_parameters_with_variability,
                                                                                self.starting_conditions_with_variability)

        solutions = []

        for v in all_vecs:
            solutions.append(extract_params_from_i0(v, self.starting_parameters_with_variability,
                                                    self.starting_conditions_with_variability))

        result = InferenceResult(problem, self.observed_trajectories,
                                 self.starting_parameters, self.starting_conditions,
                                 optimal_parameters, optimal_initial_conditions,
                                 distance_at_minimum,
                                 iterations_taken,
                                 function_calls_made,
                                 warning_flag,
                                 solutions,
                                 self._simulation)
        return result

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
        return constraints_are_satisfied(current_guess, self.constraints)


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

def write_inference_results(restart_results, t, vary, initcond_full, varyic, inferfile):
    """
    Writes inference results to output file (default name = inference.txt)

    'result' is the output from python fmin function:
     result = (optimised i0, optimised distance, no. of iterations required,
     no. of function evaluations made, warning flag - 0 if successful,
     1 or 2 if not)

    :param restart_results: list of lists [[result, mu, param, initcond]].
                            Internal list for each optimisation performed (i.e. if several random restarts selected
                            using --restart option in main menu, each set of results, sorted in order of increasing
                            distance will be present in `restart_results`). If not restarts used, there will be
                            just one internal list.
    :param t:
    :param vary:
    :param initcond_full:
    :param varyic:
    :param inferfile:
    :return:
    """
    outfile = open(inferfile, 'w')
    for i in range(len(restart_results)):
        outfile.write('Starting parameters:\t' + str(restart_results[i][2]) + '\n')
        (opt_param, opt_initconds) = extract_params_from_i0(list(restart_results[i][0][0]), zip(restart_results[i][2], vary),
                                                zip(initcond_full, varyic))
        outfile.write('Optimised parameters:\t' + str(opt_param) + '\n')
        outfile.write('Starting initial conditions:\t' + str(restart_results[i][3]) + '\n')
        outfile.write('Optimised initial conditions:\t' + str(opt_initconds) + '\n')
        if restart_results[i][0][4] == 0:
            outfile.write('Optimisation successful:\n')
            outfile.write('\tNumber of iterations: ' + str(restart_results[i][0][2]) + '\n')
            outfile.write('\tNumber of function evaluations: ' + str(restart_results[i][0][3]) + '\n')
            outfile.write('\tDistance at minimum: ' + str(restart_results[i][0][1]) + '\n\n')
        if restart_results[i][0][4] != 0:
            outfile.write('Optimisation terminated: maximum number of iterations or function evaluations exceeded.\n')
            outfile.write('\tNumber of iterations: ' + str(restart_results[i][0][2]) + '\n')
            outfile.write('\tNumber of function evaluations: ' + str(restart_results[i][0][3]) + '\n')
            outfile.write('\tDistance at minimum: ' + str(restart_results[i][0][1]) + '\n\n')


def graph(problem, opt_results, observed_trajectories, timepoints, initcond_full, vary, varyic, plottitle):
    """
    Plots graph of data vs inferred trajectories (max of 9 subplots created)

    Moment trajectories calculated using intial parameters (green line) and optimised parameters (red line),
    with the experiment data as black circles.

    :param opt_results:
    :param timepoints:
    :param initcond_full:
    :param vary:
    :param varyic:
    :param mfkoutput:
    :param plottitle:
    :param mom_index_list:
    :param moments_list:
    :return:
    """
    (opt_param, opt_initcond) = extract_params_from_i0(list(opt_results[0][0]), opt_results[2], vary, initcond_full, varyic)

    # get trajectories for optimised parameters
    simulator = Simulation(problem)
    starting_trajectories = simulator.simulate_system(opt_results[2], opt_results[3], timepoints)
    optimal_trajectories = simulator.simulate_system(opt_param, opt_initcond, timepoints)

    trajectory_lookup = { start_trajectory.description: (start_trajectory, optimal_trajectory)
                          for start_trajectory, optimal_trajectory in zip(starting_trajectories, optimal_trajectories)}
    # Plot figure (starting vs optimised trajectories, plus experimental data)
    fig = plt.figure()

    # Try to guess the best way to split the plot
    rows = int(sqrt(len(observed_trajectories)))
    columns = int(sqrt(len(observed_trajectories)))
    while rows*columns < len(observed_trajectories):
        rows += 1

    for i, observed_trajectory in enumerate(observed_trajectories):
        ax = plt.subplot(rows, columns, i+1)

        ax.plot(observed_trajectory.timepoints, observed_trajectory.values, color='k', linestyle='None', marker='.',
                label='Observed')
        plt.xlabel('time')
        plt.ylabel(observed_trajectory.description)

        starting_trajectory, optimal_trajectory = trajectory_lookup[observed_trajectory.description]

        ax.plot(starting_trajectory.timepoints, starting_trajectory.values, color='g', label='Starting')
        ax.plot(optimal_trajectory.timepoints, optimal_trajectory.values, color='r', label='Optimal')
        ax.yaxis.set_major_locator(MaxNLocator(5))


    plt.legend(bbox_to_anchor=(1.0, -0.5))
    plt.tight_layout()
    fig.suptitle(plottitle)
    plt.show()