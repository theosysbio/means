"""
Functions used for inference using parametric distribution models

Nelder-Mead simplex algorithm used to minimise negative log likelihood
i.e. -ln L(x|theta) where theta is the set of parameters to vary
(selected kinetic parameters and initial conditions).
Likelihood (L) is calculated by approximating the distribution
by the chosen density function (--pdf=normal, gamma, lognormal), and
assuming independence between timepoints and sample data points.

For each set of parameters (i0), the mean and variance trajectories
are simulated by solving the MFK equations.  These values are used
to sum the log likelihood over each time/data point.
"""
from collections import namedtuple
import os
import sys
from math import sqrt, log, pi, lgamma

from scipy.optimize import fmin

from ode_problem import Moment
from simulate import Simulation
from sumsq_infer import make_i0, i0_to_test, parse_experimental_data_file
import numpy as np

def eval_density(mean, var, x, distribution):
    """
    Calculates gamma/lognormal/normal pdf given mean variance, x
    where x is the experimental species number measured at a particular timepoint. Returns ln(pdf)
    :param mean: mean
    :param var: variance
    :param x: experimental species number measured at a particular timepoint
    :param distribution: distribution to consider. Either 'gamma', 'normal' and 'lognormal'
    :return: normal log of the pdf
    """
    if distribution == 'gamma':
        b = var / mean
        a = mean / b
        logpdf = (a - 1) * log(x) - (x / b) - a * log(b) - lgamma(a)

    elif distribution == 'normal':
        logpdf = -(x - mean) ** 2 / (2 * var) - log(sqrt(2 * pi * var))

    elif distribution == 'lognormal':
        logpdf = -(log(x) - mean) ** 2 / (2 * var) - log(x * sqrt(2 * pi * var))

    return logpdf

#####################################################################
# Optimise function, used for parameter inference (minimizes
# the distance/cost funtion using a simplex algorithm)
# Arguments: 
# param, vary      kinetic parameters, and list to indicate which to infer
# initcond,varyic  initial conditions, " "
# sample           name of experimental data file
# cfile            name of C library (specified by --library)
# mfkoutput        name of file produced by MFK (specified by --ODEout)
# distribution     parametric model (specified by --pdf)
######################################################################     

MeanVariance = namedtuple('MeanVariance', ['mean', 'variance'])
def _compile_mean_variance_lookup(trajectories):
    means = {}
    variances = {}

    for trajectory in trajectories:
        description = trajectory.description
        if not isinstance(description, Moment):
            continue
        moment = description
        if moment.order == 1:
            # TODO: np.where looks nasty here maybe we could use something else
            # Currently there isn't any function in numpy to find first element in the list
            # see https://github.com/numpy/numpy/issues/2269
            species_id = np.where(moment.n_vector == 1)[0][0]
            means[species_id] = trajectory.values
        elif moment.order == 2 and not moment.is_mixed:
            species_id = np.where(moment.n_vector == 2)[0][0]
            variances[species_id] = trajectory.values

    combined_lookup = {}
    for species, mean in means.iteritems():
        combined_lookup[species] = MeanVariance(mean, variances[species])

    return combined_lookup

def optimise(problem, param, vary, initcond, varyic, limits, sample, distribution):
    """
    Optimise function, used for parameter inference (minimizes the distance/cost funtion using a simplex algorithm)
    :param problem:
    :type problem: ODEProblem
    :param param: kinetic parameters
    :param vary: list which kinetic parameters to infer
    :param initcond: initial conditions
    :param varyic: list of which initial conditions to vary
    :param limits:
    :param sample: name of experimental data file
    :param distribution: parametric model (specified by `--pdf`)
    :return:
    """
    i0 = make_i0(param, vary, initcond, varyic)    # create initial i0

    # Get required info from MFK output file, extend initcond with default 
    # value of 0 if only some of the initial moment values are specified

    number_of_species = problem.number_of_species
    number_of_equations = problem.number_of_equations
    if len(initcond) != number_of_equations:
        initcond += ([0] * (number_of_equations - len(initcond)))

    ######################################################################
    # Evaluates distance (cost) function for current set of values in i0
    #
    # At each iteration, this function is called by fmin and calculated 
    # using the current values in i0.  Returned value (dist) is minimised
    # by varying i0
    #
    # Distance is sum of -log(likelihood) assuming independence between
    # time/data points
    #####################################################################

    def distance(i0, param, vary, initcond, varyic, observed_trajectories, observed_timepoints, distribution):

        # value returned if parameters or means < 0 or outside limits
        max_dist = 1.0e10

        # parameters to solve for
        (test_param, test_initcond) = i0_to_test(i0, param, vary, initcond, varyic)

        # Check parameters are positive, and within limits (if --limit used)
        if any(i < 0 for i in test_param):     # parameters cannot be negative
            return max_dist
        if any(j < 0 for j in test_initcond[0:number_of_species]):
            return max_dist      # disallow negative means

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


        simulator = Simulation(problem, postprocessing='LNA' if problem.method=='LNA' else None)
        simulated_timepoints, simulated_trajectories = simulator.simulate_system(test_param, test_initcond,
                                                                                 observed_timepoints)

        mean_variance_lookup = _compile_mean_variance_lookup(simulated_trajectories)

        # get moment expansion result with current parameters
        log_likelihood = 0

        ###############################################################
        # If a parametric distribution used..
        ################################################################

        for trajectory in observed_trajectories:
            moment = trajectory.description
            assert(isinstance(moment, Moment))
            assert(moment.order == 1)

            species = np.where(moment.n_vector == 1)[0][0]
            mean_variance = mean_variance_lookup[species]
            if (mean_variance.mean < 0).any() or (mean_variance.variance < 0).any():
                return max_dist

            for tp, value in enumerate(trajectory.values):
                if np.isnan(value):
                    continue

                mean = mean_variance.mean[tp]
                variance = mean_variance.variance[tp]

                # can't calculate parametric likelihoods for zero variance, so pass
                if variance == 0:
                    continue

                log_likelihood += eval_density(mean, variance, value, distribution)

        dist = -log_likelihood
        y_list.append(dist)
        i0_list.append(i0[0])
        return dist

    # callback: function called after each iteration (each iteration will involve several
    # distance function evaluations).  Use this to save data after each iteration if wanted.
    # x is the current i0 returned after that iteration.    

    def my_callback(x):
        it_param.append(x[0])
        it_no.append(len(it_param))

    # create lists to collect data at each iteration (used during testing)
    y_list = []
    i0_list = []
    it_no = []
    it_param = []
    it_dist = []

    # read sample data from file and get indices for mean/variances in CVODE output
    (observed_timepoints, observed_trajectories) = parse_experimental_data_file(sample)
    #(mom_index_list,moments_list) = mom_indices(mfkoutput, mom_names)

    # minimise defined distance function, with provided starting parameters
    result = fmin(distance, i0,
                  args=(param, vary, initcond, varyic, observed_trajectories, observed_timepoints,distribution),
                  ftol=0.000001, disp=0, full_output=True, callback=my_callback)

    return result, observed_timepoints, observed_trajectories, initcond


