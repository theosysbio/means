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

from means.util.decorators import memoised_property
from means.inference.gamma_infer import _distribution_distance, SUPPORTED_DISTRIBUTIONS
from means.approximation.ode_problem import Moment
from means.simulation.simulate import Simulation, NP_FLOATING_POINT_PRECISION, Trajectory

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



def i0_to_test(only_variable_parameters, parameters_with_variability, initial_conditions_with_variability):
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

class ParameterInference(object):

    __problem = None
    __starting_parameters_with_variability = None
    __starting_conditions_with_variability = None
    __constraints = None
    __observed_timepoints = None
    __observed_trajectories = None
    _method = None

    def __init__(self, problem, starting_parameters_with_variability, starting_conditions_with_variability,
                 constraints, observed_timepoints, observed_trajectories, method='sum_of_squares'):
        """
        :param problem: ODEProblem to infer data for
        :type problem: ODEProblem
        :param starting_parameters_with_variability: List of tuples (value, is_variable) for each parameter in ODEProblem
                                     specification.
                                    - value is the starting value for that parameter
                                    - is_variable is a boolean True/False signifying whether the constant can be varied
                                      during inference
        :param starting_conditions_with_variability: similar list of tuples as `starting_parameters`,
                                    but for each of the LHS equations in the problem
        :param constraints:         List of tuples (lower_bound, upper_bound) for each of the variable parameters.
                                    TODO:  Maybe a better way exists to specify these constraints.
                                    This is a bit weird way: we specify only variable ones.
                                    I am thinking somewhat about a container class Parameter(value, variable, constraints)
                                    Can be none if no constraints present, similarly each of the bounds can be None,
                                    if unspecified
        :param observed_timepoints:   Timepoints for which we have data for
        :param observed_trajectories: A list of `Trajectory` objects containing observed data values.
        :param method: Method of calculating the data fit. Currently supported values are
              - 'sum_of_squares' -  min sum of squares optimisation
              - 'gamma' - maximum likelihood optimisation assuming gamma distribution
              - 'normal'- maximum likelihood optimisation assuming normal distribution
              - 'lognormal' - maximum likelihood optimisation assuming lognormal distribution
        """
        self.__problem = problem

        assert(len(starting_parameters_with_variability) == len(problem.constants))
        assert(len(starting_conditions_with_variability) == problem.number_of_equations)

        self.__starting_parameters_with_variability = starting_parameters_with_variability
        self.__starting_conditions_with_variability = starting_conditions_with_variability

        assert(constraints is None or len(constraints) == len(filter(lambda x: x[1],
                                                                     starting_parameters_with_variability +
                                                                     starting_conditions_with_variability)))
        self.__constraints = constraints

        self.__observed_timepoints = observed_timepoints
        self.__observed_trajectories = observed_trajectories

        self._method = method

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
        simulation_type = problem.method

        timepoints_to_simulate = self.observed_timepoints

        _distance_between_trajectories_function = self._distance_between_trajectories_function
        def distance(current_guess):
            if not self._constraints_are_satisfied(current_guess):
                return MAX_DIST

            current_parameters, current_initial_conditions = i0_to_test(current_guess,
                                                                        starting_parameters_with_variability,
                                                                        starting_conditions_with_variability)

            if some_params_are_negative(problem, current_parameters, current_initial_conditions):
                return MAX_DIST

            simulator = Simulation(self.problem, postprocessing='LNA' if simulation_type == 'LNA' else None)
            simulated_timepoints, simulated_trajectories = simulator.simulate_system(current_parameters,
                                                                                     current_initial_conditions,
                                                                                     timepoints_to_simulate)

            dist = _distance_between_trajectories_function(simulated_trajectories, observed_trajectories_lookup)
            return dist

        result = fmin(distance, initial_guess, ftol=FTOL, disp=0, full_output=True)

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
        return [x[0] for x in self.starting_conditions_with_variability]

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
        (opt_param, opt_initconds) = i0_to_test(list(restart_results[i][0][0]), zip(restart_results[i][2], vary),
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
    (opt_param, opt_initcond) = i0_to_test(list(opt_results[0][0]), opt_results[2], vary, initcond_full, varyic)

    # get trajectories for optimised parameters
    simulator = Simulation(problem, postprocessing='LNA' if problem.method == 'LNA' else None)
    __, starting_trajectories = simulator.simulate_system(opt_results[2], opt_results[3], timepoints)
    __, optimal_trajectories = simulator.simulate_system(opt_param, opt_initcond, timepoints)

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