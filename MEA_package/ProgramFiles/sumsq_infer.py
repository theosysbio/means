#################################################################
# Functions used for parameter inference.  Selected rate constants
# and initial conditions are varied to minimise the cost function.
# Minimization uses the Nelder-Mead simplex algorithm (python fmin).
# The default cost function calculates the distance (sum of squared
# differences) between the sample data moments and moments 
# calculated using MFK at each of the specified timepoints.
#################################################################
import numpy as np
from sympy import Matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import fmin
from CVODE import CVODE
import re
from math import factorial
from ode_problem import Moment
from simulate import Simulation, NP_FLOATING_POINT_PRECISION, Trajectory


def make_i0(param, vary, initcond, varyic):
    """
    Creates a list of variables to infer, based on the values in vary/varyic (0=fixed, 1=optimised).

    This should contain all variables that are varied as it would be passed to optimisation method.

    :param param: list of starting values for kinetic parameters
    :param vary: list to identify which values in `param` to vary during inference (0=fixed, 1=optimise)
    :param initcond: list of starting values (i.e. at t0) for moments
    :param varyic: list to identify which values in `initcond` to vary (0=fixed, 1=optimise)
    :return: i0 (which is passed to the fmin minimisation function)
    """
    i0 = []
    for i in range(0, len(vary)):
        if vary[i] == 1:
            i0.append(param[i])
    for j in range(0, len(varyic)):
        if varyic[j] == 1:
            i0.append(initcond[j])
    return i0


def i0_to_test(i0, param, vary, initcond, varyic):
    """
    Used within the distance/cost function to create the current kinetic parameter and initial condition vectors
    to be used during that interaction, using current values in i0.

    This function takes i0 and complements it with additional information from variables that we do not want to vary
    so the simulation function could be run and values compared.

    :param i0: `i0` list returned from `make_i0`
    :param param: list of starting values for kinetic parameters
    :param vary: list to identify which values in `param` to vary during inference (0=fixed, 1=optimise)
    :param initcond: list of starting values (i.e. at t0) for moments
    :param varyic: list to identify which values in `initcond` to vary (0=fixed, 1=optimise)
    :return:
    """
    test_param = param[:]
    test_initcond = initcond[:]
    i0_index = 0
    for i in range(0, len(vary)):
        if vary[i] == 1:
            test_param[i] = i0[i0_index]
            i0_index += 1
    for j in range(0, len(varyic)):
        if varyic[j] == 1:
            test_initcond[j] = i0[i0_index]
            i0_index += 1

    return (test_param, test_initcond)


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


def optimise(param, vary, initcond, varyic, limits, sample, problem):
    """
    Optimise function, that is used for parameter inference.

    This function minimises the distance/cost function using simplex algorithm.

    :param param: kinetic parameters
    :param vary: list of parameters which are to be infered
    :param initcond: initial conditions
    :param varyic: " " TODO: this is the original test, does this mean empty?
    :param limits: constrains allowed values for parameters and initial conditions (list with entry in form lower,upper)
                   for each parameter/initial condition set in `timeparam` file.
    :param sample: name of the experimental data file
    :param problem: The specified ODE problem to optimise
    :type problem: ODEProblem
    :return:
    """
    i0 = make_i0(param, vary, initcond, varyic)        # create initial i0

    # Get required information from the MFK or LNA output file 

    simulation_type = problem.method            # simtype = MFK or LNA
    number_of_species = problem.number_of_species


    number_of_equations = problem.number_of_equations

    # If starting values not specified for all moments, set remainder to 0
    if len(initcond) != number_of_equations:
        initcond += ([0] * (number_of_equations - len(initcond)))

    def distance(i0, param, vary, initcond, varyic, observed_trajectories_lookup, observed_timepoint):
        """
        Evaluates distance (cost) function for current set of values in i0.

        At each iteration, this function is called by fmin and calculated using the current values in i0.
        Returned value (dist) is minimised by varying i0.

        Distance is the sum of squared differences between the sample data, and the corresponding values
        simulated using either MFK or LNA with the current parameter sets.

        :param i0:
        :param param:
        :param vary:
        :param initcond:
        :param varyic:
        :param observed_trajectories_lookup: a dictionary in form {Moment: trajectory}
                                             where trajectory is the observed trajectories
        :param observed_timepoints: timepoints (from sample data file)
        :param observed_moments_index_list: the indices of the corresponding moments in simulated trajectories produced by CVODE
        :return:
        """

        # value returned if parameters, means or variances < 0
        max_dist = 1.0e10

        # creates lists of parameters (test_param and test_initcond) for that iteration

        (test_param, test_initcond) = i0_to_test(i0, param, vary, initcond, varyic)

        simulator = Simulation(problem, postprocessing='LNA' if simulation_type=='LNA' else None)
        simulated_timepoints, simulated_trajectories = simulator.simulate_system(test_param, test_initcond,
                                                                                 observed_timepoints)

        # Check if parameters/initconds are within allowed bounds (if --limit used)
        # and return max_dist if outside these ranges
        # FIXME: Maybe worth solving the ODEs (i.e. obtaining `test_soln` *after* checking for limits ?

        if limits is not None:
            for a in range(0, len(i0)):
                l_limit = limits[a][0]
                u_limit = limits[a][1]
                if l_limit != 'N':
                    if i0[a] < l_limit:
                        return max_dist
                if u_limit != 'N':
                    if i0[a] > u_limit:
                        return max_dist

        # If MFK used, distance summed over all timepoints/moments contained 
        # in sample data file

        if simulation_type == 'MEA':
            if any(i < 0 for i in test_param):   # disallow negative kinetic parameters
                return max_dist

            # calculate number of var/covar terms
            #nVar_Covar = factorial(nspecies+1)/(factorial(2)*factorial(nspecies-2))
            #if any(j<0 for j in test_initcond[0:nspecies+nVar_Covar]):
            # Fixme: these should probably be in LNA?
            if any(j < 0 for j in test_initcond[0:number_of_species]):
                return max_dist              # disallow negative means/variance/covariance

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

        # If LNA used...

        # if simulation_type == 'LNA':
        #     tmu = [0] * number_of_species
        #     mu_t = [0] * len(observed_timepoints)
        #     for i in range(0, number_of_species):
        #         mu_i = [0] * len(observed_timepoints)
        #         for j in range(len(observed_timepoints)):
        #             if i == 0:
        #                 V = Matrix(number_of_species, number_of_species, lambda k, l: 0)
        #                 for v in range(2 * number_of_species):
        #                     V[v] = simulated_trajectories[j, v + number_of_species]
        #                 mu_t[j] = np.random.multivariate_normal(simulated_trajectories[j, 0:number_of_species], V)
        #             mu_i[j] = mu_t[j][i]
        #         tmu[i] = mu_i
        #     dist = 0
        #     for species in range(0, len(observed_trajectories)):
        #         for i_timepoint in range(0, len(observed_timepoints)):
        #             dist += (observed_trajectories[species][i_timepoint] - tmu[species][i_timepoint]) ** 2

        # Returns distance (dist), (and saves current i0 and dist to list)
        y_list.append(dist)
        i0_list.append(i0[0])
        return dist

    def my_callback(x):
        """
        Callback function called after each iteration (each iteration will involve several distance function
        evaluations). Use this to save data after each iteration if wanted.
        :param x: Current i0 returned after that iteration
        :return:
        """
        it_param.append(x)
        it_no.append(len(it_param))
        #it_dist.append(y_list[-1])
        #print x
        #print y_list[-1]


    # create lists to collect data at each iteration (use with my_callback if wanted)
    y_list = []
    i0_list = []
    it_no = []
    it_param = []
    it_dist = []

    # read sample data from file to get required information to pass to fmin
    observed_timepoints, observed_trajectories = parse_experimental_data_file(sample)
    observed_trajectories_lookup = {trajectory.description: trajectory for trajectory in observed_trajectories}

    # minimise defined distance function, with provided starting parameters
    result = fmin(distance, i0, args=(param, vary, initcond, varyic, observed_trajectories_lookup,
                                      observed_timepoints),
                  ftol=0.000001, disp=0, full_output=True, callback=my_callback)

    return result, observed_timepoints, observed_trajectories, initcond

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
        (opt_param, opt_initconds) = i0_to_test(list(restart_results[i][0][0]), restart_results[i][2], vary,
                                                initcond_full, varyic)
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


def graph(opt_results, mu, t, lib, initcond_full, vary, varyic, mfkoutput, plottitle, mom_index_list, moments_list):
    """
    Plots graph of data vs inferred trajectories (max of 9 subplots created)

    Moment trajectories calculated using intial parameters (green line) and optimised parameters (red line),
    with the experiment data as black circles.

    :param opt_results:
    :param t:
    :param lib:
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
    opt_soln = CVODE(lib, t, opt_initcond, opt_param)
    opt_mu = [opt_soln[:, i] for i in range(0, len(initcond_full))]

    # get trajectories for starting parameters
    start_soln = CVODE(lib, t, opt_results[3], opt_results[2])
    start_mu = [start_soln[:, i] for i in range(0, len(initcond_full))]

    # Plot figure (starting vs optimised trajectories, plus experimental data)
    fig = plt.figure()
    plot_list = []

    # Allow for missing timepoints in experimental data
    def check_partial_data(times, traj):
        t_list = []
        traj_list = []
        for a in range(len(traj)):
            if traj[a] != 'N':
                t_list.append(times[a])
                traj_list.append(traj[a])
        return (t_list, traj_list)

    for i in mom_index_list:
        new_plot = False
        if i not in plot_list:
            plot_list.append(i)
            new_plot = True

        if len(plot_list) < 10:
            j = plot_list.index(i)
            ax = plt.subplot(4, 3, j + 1)
            (t_list, traj_list) = check_partial_data(t, mu[i])

            # if no plot exists for that moment create subplot, else add to existing subplot 
            if new_plot == True:
                ax.plot(t_list, traj_list, color='k', linestyle='None', marker='.', label='data')
                plt.xlabel('t')
                plt.ylabel(moments_list[mom_index_list[i]])
                ax.plot(t, opt_mu[mom_index_list[i]], color='r', label='optimised')
                ax.plot(t, start_mu[mom_index_list[i]], color='g', label='starting values')
            else:
                ax.plot(t_list, traj_list, color='k', linestyle='None', marker='.')
            ax.yaxis.set_major_locator(MaxNLocator(5))

    ax.legend(bbox_to_anchor=(1.0, -0.5))
    plt.tight_layout()
    fig.suptitle(plottitle)
    plt.show()