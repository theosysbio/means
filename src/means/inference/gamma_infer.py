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

from scipy.special import gammaln
import numpy as np

from means.approximation.ode_problem import Moment
from means.simulation.simulate import NP_FLOATING_POINT_PRECISION

SUPPORTED_DISTRIBUTIONS = {'gamma', 'normal', 'lognormal'}

def eval_density(means, variances,observed_values, distribution):
    """
    Calculates gamma/lognormal/normal pdf given mean variance, x
    where x is the experimental species number measured at a particular timepoint. Returns ln(pdf)
    :param mean: mean
    :param var: variance
    :param observed_values: experimental species number measured at a particular timepoint
    :param distribution: distribution to consider. Either 'gamma', 'normal' and 'lognormal'
    :return: normal log of the pdf
    """
    means = np.array(means, dtype=NP_FLOATING_POINT_PRECISION)
    variances = np.array(variances, dtype=NP_FLOATING_POINT_PRECISION)
    observed_values = np.array(observed_values, dtype=NP_FLOATING_POINT_PRECISION)

    # Remove data about unobserved datapoints
    means = means[~np.isnan(observed_values)]
    variances = variances[~np.isnan(observed_values)]
    observed_values = observed_values[~np.isnan(observed_values)]

    # Remove data for when variance is zero as we cannot estimate distributions that way
    non_zero_varianes = ~(variances == 0)
    means = means[non_zero_varianes]
    variances = variances[~(variances == 0)]
    observed_values = observed_values[non_zero_varianes]

    if distribution == 'gamma':
        b = variances / means
        a = means / b

        log_observed_values = np.log(observed_values)
        log_density = (a - 1.0) * log_observed_values - (observed_values / b) - a * np.log(b) - gammaln(a)
    elif distribution == 'normal':
        log_density = -(observed_values - means) ** 2 / (2 * variances) - np.log(np.sqrt(2 * np.pi * variances))

    elif distribution == 'lognormal':
        log_density = -(np.log(observed_values) - means) ** 2 / (2 * variances) - np.log(observed_values * np.sqrt(2 * np.pi * variances))
    else:
        raise ValueError('Unsupported distribution {0!r}'.format(distribution))

    total_log_density = np.sum(log_density)
    return total_log_density

_MeanVariance = namedtuple('_MeanVariance', ['mean', 'variance'])
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
        combined_lookup[species] = _MeanVariance(mean, variances[species])

    return combined_lookup

def _distribution_distance(simulated_trajectories, observed_trajectories_lookup, distribution):

    mean_variance_lookup = _compile_mean_variance_lookup(simulated_trajectories)

    # get moment expansion result with current parameters
    log_likelihood = 0

    ###############################################################
    # If a parametric distribution used..
    ################################################################

    for trajectory in observed_trajectories_lookup.itervalues():
        moment = trajectory.description
        assert(isinstance(moment, Moment))
        assert(moment.order == 1)

        species = np.where(moment.n_vector == 1)[0][0]
        mean_variance = mean_variance_lookup[species]
        if (mean_variance.mean < 0).any() or (mean_variance.variance < 0).any():
            return float('inf')

        term = eval_density(mean_variance.mean, mean_variance.variance, trajectory.values, distribution)
        log_likelihood += term

    dist = -log_likelihood
    return dist


