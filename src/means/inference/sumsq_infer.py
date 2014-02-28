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
from means.inference.hypercube import hypercube
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
        :rtype: list[tuple]|None
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
        if self.solutions is None:
            return []

        timepoints = self.observed_trajectories[0].timepoints
        return map(lambda x: self._simulation.simulate_system(x[0], x[1], timepoints), self.solutions)

    def plot(self, plot_intermediate_solutions=True, filter_plots_function=None, legend=True,
             kwargs_observed_data=None, kwargs_starting_trajectories=None, kwargs_optimal_trajectories=None,
             kwargs_intermediate_trajectories=None):
        """
        Plot the inference result.

        :param plot_intermediate_solutions: plot the trajectories resulting from the intermediate solutions as well
        :param filter_plots_function: A function that takes a trajectory object and returns True if it should be
                                      plotted and false if not. None plots all available trajectories
        :param legend: Whether to draw the legend or not
        :param kwargs_observed_data: Kwargs to be passed to the ``trajectory.plot`` function for the observed data
        :param kwargs_starting_trajectories: kwargs to be passed to the ``trajectory.plot`` function for the starting
                                             trajectories
        :param kwargs_optimal_trajectories: kwargs to be passed to the ``trajectory.plot`` function for the optimal
                                            trajectories
        :param kwargs_intermediate_trajectories: kwargs to be passed to the ``trajectory.plot`` function for the
                                            intermediate trajectories
        """
        from matplotlib import pyplot as plt
        if filter_plots_function is None:
            filter_plots_function = lambda x: True

        observed_trajectories = self.observed_trajectories

        starting_trajectories = self.starting_trajectories
        optimal_trajectories = self.optimal_trajectories

        if plot_intermediate_solutions:
            intermediate_trajectories_list = self.intermediate_trajectories
        else:
            intermediate_trajectories_list = []

        def initialise_default_kwargs(kwargs, default_data):
            if kwargs is None:
                kwargs = {}

            for key, value in default_data.iteritems():
                if key not in kwargs:
                    kwargs[key] = value

            return kwargs

        trajectories_by_description = {}
        kwargs_observed_data = initialise_default_kwargs(kwargs_observed_data, {'label': "Observed data", 'marker': '+',
                                                                                'color': 'black', 'linestyle': 'None'})

        kwargs_optimal_trajectories = initialise_default_kwargs(kwargs_optimal_trajectories,
                                                                {'label': "Optimised Trajectory", 'color': 'blue'})

        kwargs_starting_trajectories = initialise_default_kwargs(kwargs_starting_trajectories,
                                                                 {'label': "Starting trajectory", 'color': 'green'})

        kwargs_intermediate_trajectories = initialise_default_kwargs(kwargs_intermediate_trajectories,
                                                                     {'label': 'Intermediate Trajectories',
                                                                      'alpha': 0.1, 'color': 'cyan'}
                                                                     )

        for trajectory in observed_trajectories:
            if not filter_plots_function(trajectory):
                continue

            try:
                list_ = trajectories_by_description[trajectory.description]
            except KeyError:
                list_ = []
                trajectories_by_description[trajectory.description] = list_

            list_.append((trajectory, kwargs_observed_data))

        for trajectory in starting_trajectories:
            if not filter_plots_function(trajectory):
                continue

            try:
                list_ = trajectories_by_description[trajectory.description]
            except KeyError:
                list_ = []
                trajectories_by_description[trajectory.description] = list_

            list_.append((trajectory, kwargs_starting_trajectories))

        seen_intermediate_trajectories = set()
        for i, intermediate_trajectories in enumerate(intermediate_trajectories_list):
            for trajectory in intermediate_trajectories:
                if not filter_plots_function(trajectory):
                    continue

                seen = trajectory.description in seen_intermediate_trajectories
                kwargs = kwargs_intermediate_trajectories.copy()
                # Only set label once
                if not seen:
                    seen_intermediate_trajectories.add(trajectory.description)
                else:
                    kwargs['label'] = ''

                try:
                    list_ = trajectories_by_description[trajectory.description]
                except KeyError:
                    list_ = []

                trajectories_by_description[trajectory.description] = list_
                list_.append((trajectory, kwargs))


        for trajectory in optimal_trajectories:
            if not filter_plots_function(trajectory):
                continue

            try:
                list_ = trajectories_by_description[trajectory.description]
            except KeyError:
                list_ = []
                trajectories_by_description[trajectory.description] = list_

            list_.append((trajectory, kwargs_optimal_trajectories))

        for description, trajectories_list in trajectories_by_description.iteritems():
            if len(trajectories_by_description) > 1:
                plt.figure()
            plt.title(description)

            for trajectory, kwargs in trajectories_list:
                trajectory.plot(**kwargs)

            if legend:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


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
                   ('solutions', data.solutions),
                   ('simulation', data._simulation)]

        return dumper.represent_mapping(cls.yaml_tag, mapping)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.problem == other.problem and self.observed_trajectories == other.observed_trajectories\
            and self.starting_parameters == other.starting_parameters \
            and self.starting_initial_conditions == other.starting_initial_conditions \
            and self.optimal_parameters == other.optimal_parameters \
            and self.optimal_initial_conditions == other.optimal_initial_conditions \
            and self.distance_at_minimum == other.distance_at_minimum \
            and self.iterations_taken == other.iterations_taken \
            and self.function_calls_made == other.function_calls_made \
            and self.warning_flag == other.warning_flag \
            and self.solutions == other.solutions \
            and self._simulation == other._simulation

class InferenceResultsCollection(object):
    __inference_results = None

    def __init__(self, inference_results):
        self.__inference_results = sorted(inference_results, key=lambda x: x.distance_at_minimum)

    @property
    def results(self):
        """

        :return: The results of performed inferences
        :rtype: list[:class:`InferenceResult`]
        """
        return self.__inference_results

    @property
    def number_of_results(self):
        return len(self.results)

    @property
    def best(self):
        return self.__inference_results[0]

    def __unicode__(self):
        return u"""
        {self.__class__!r}

        Number of inference results in collection: {self.number_of_results}
        Best:
        {self.best!r}
        """.format(self=self)

    def __str__(self):
        return unicode(self).encode("utf8")

    def __repr__(self):
        return str(self)


    def plot(self):
        from matplotlib import pyplot as plt

        trajectory_descriptions = [x.description for x in self.results[0].starting_trajectories]


        # Plot in reverse order so the best one is always on top
        reversed_results = list(reversed(self.results))
        # Plot all but last one (as the alpha will change)

        # Let's make worse results fade
        alpha = 0.2

        for description in trajectory_descriptions:
            plt.figure()
            plt.title(description)
            f = lambda trajectory: trajectory.description == description
            first = True
            for result in reversed_results[:-1]:
                if first:
                    label_starting = 'Alternative Starting Trajectories'
                    label_optimal = 'Alternative Optimised Trajectories'
                    first = False
                else:
                    label_starting = ''
                    label_optimal = ''

                result.plot(filter_plots_function=f,
                            legend=False, kwargs_starting_trajectories={'alpha': alpha, 'label': label_starting},
                            kwargs_optimal_trajectories={'alpha': alpha, 'label': label_optimal},
                            # Do not draw observed data, it is the same for all
                            kwargs_observed_data={'label': '', 'alpha': 0},
                            plot_intermediate_solutions=False)

            self.best.plot(filter_plots_function=f, plot_intermediate_solutions=False,
                           kwargs_starting_trajectories={'label': 'Best Starting Trajectory'},
                           kwargs_optimal_trajectories={'label': 'Best Optimised Trajectory'})

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
                raise ValueError('Invalid range provided: expected ``(min_value, max_value)`` got {0!r}'.format(item))
            if len(item) != 2 or item[0] > item[1]:
                raise ValueError('Invalid range provided: expected: ``(min_value, max_value)`` got {0!r}'.format(item))

            validated_range.append(item)

        return validated_range


    def __init__(self, problem, number_of_samples,
                 starting_parameter_ranges, starting_conditions_ranges,
                 variable_parameters, observed_trajectories, method='sum_of_squares',
                 return_intermediate_solutions=False,
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

    @memoised_property
    def _inference_objects(self):

        full_list_of_ranges = self.starting_parameter_ranges[:] + self.starting_conditions_ranges[:]
        variables_collection = hypercube(self.number_of_samples, full_list_of_ranges)

        inference_objects = []
        for variables in variables_collection:
            starting_parameters = variables[:len(self.starting_parameter_ranges)]
            starting_conditions = variables[len(self.starting_parameter_ranges):]

            inference_objects.append(ParameterInference(self.problem,
                                                        starting_parameters,
                                                        starting_conditions,
                                                        self.variable_parameters,
                                                        self.observed_trajectories,
                                                        method=self.method,
                                                        return_intermediate_solutions=self._return_intermediate_solutions,
                                                        **self.simulation_kwargs))

        return inference_objects

    def infer(self):

        results = [x.infer() for x in self._inference_objects]
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





class ParameterInference(object):

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

        result = fmin(distance, initial_guess, ftol=FTOL, disp=0, full_output=True,
                      retall=self._return_intermediate_solutions)

        if self._return_intermediate_solutions:
            optimised_data, distance_at_minimum, iterations_taken, function_calls_made, warning_flag, all_vecs = result
        else:
            optimised_data, distance_at_minimum, iterations_taken, function_calls_made, warning_flag = result
            all_vecs = None

        optimal_parameters, optimal_initial_conditions = extract_params_from_i0(optimised_data,
                                                                                self.starting_parameters_with_variability,
                                                                                self.starting_conditions_with_variability)

        if all_vecs is not None:
            solutions = []

            for v in all_vecs:
                solutions.append(extract_params_from_i0(v, self.starting_parameters_with_variability,
                                                        self.starting_conditions_with_variability))
        else:
            solutions=None

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