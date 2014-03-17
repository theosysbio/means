from collections import namedtuple
import numpy as np
from means.core import Moment
from means.simulation.solvers import NP_FLOATING_POINT_PRECISION
from scipy.special import gammaln

def _supported_distances_lookup():
    return {'sum_of_squares': sum_of_squares,
            'gamma': gamma,
            'normal': normal,
            'lognormal': lognormal}

def get_distance_function(distance):
    """
    Returns the distance function from the string name provided

    :param distance: The string name of the distributions
    :return:
    """
    # If we provided distance function ourselves, use it
    if callable(distance):
        return distance
    try:
        return _supported_distances_lookup()[distance]
    except KeyError:
        raise KeyError('Unsupported distance function {0!r}'.format(distance.lower()))

def sum_of_squares(simulated_trajectories, observed_trajectories_lookup):
    """
    Returns the sum-of-squares distance between the simulated_trajectories and observed_trajectories

    :param simulated_trajectories: Simulated trajectories
    :type simulated_trajectories: list[:class:`means.simulation.Trajectory`]
    :param observed_trajectories_lookup: A dictionary of (trajectory.description: trajectory) of observed trajectories
    :type observed_trajectories_lookup: dict
    :return: the distance between simulated and observed trajectories
    :rtype: float
    """
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

def gamma(simulated_trajectories, observed_trajectories_lookup):
    """
    Returns the negative log-likelihood of the observed trajectories assuming a gamma distribution
    on the simulated trajectories values

    :param simulated_trajectories: Simulated trajectories
    :type simulated_trajectories: list[:class:`means.simulation.Trajectory`]
    :param observed_trajectories_lookup: A dictionary of (trajectory.description: trajectory) of observed trajectories
    :type observed_trajectories_lookup: dict
    :return:
    """
    return _distribution_distance(simulated_trajectories, observed_trajectories_lookup, 'gamma')

def normal(simulated_trajectories, observed_trajectories_lookup):
    """
     Returns the negative log-likelihood of the observed trajectories assuming a normal distribution
    on the simulated trajectories values

    :param simulated_trajectories: Simulated trajectories
    :type simulated_trajectories: list[:class:`means.simulation.Trajectory`]
    :param observed_trajectories_lookup: A dictionary of (trajectory.description: trajectory) of observed trajectories
    :type observed_trajectories_lookup: dict
    :return:
    """
    return _distribution_distance(simulated_trajectories, observed_trajectories_lookup, 'normal')

def lognormal(simulated_trajectories, observed_trajectories_lookup):
    """
     Returns the negative log-likelihood of the observed trajectories assuming a log-normal distribution
    on the simulated trajectories values

    :param simulated_trajectories: Simulated trajectories
    :type simulated_trajectories: list[:class:`means.simulation.Trajectory`]
    :param observed_trajectories_lookup: A dictionary of (trajectory.description: trajectory) of observed trajectories
    :type observed_trajectories_lookup: dict
    :return:
    """
    return _distribution_distance(simulated_trajectories, observed_trajectories_lookup, 'lognormal')

_MeanVariance = namedtuple('_MeanVariance', ['mean', 'variance'])

def _distribution_distance(simulated_trajectories, observed_trajectories_lookup, distribution):
    """
    Returns the distance between the simulated and observed trajectory, w.r.t. the assumed distribution

    :param simulated_trajectories: Simulated trajectories
    :type simulated_trajectories: list[:class:`means.simulation.Trajectory`]
    :param observed_trajectories_lookup: A dictionary of (trajectory.description: trajectory) of observed trajectories
    :type observed_trajectories_lookup: dict
    :param distribution: Distribution to use. See :func:`_eval_density` for the list of available distributions
    :return:
    """

    mean_variance_lookup = _compile_mean_variance_lookup(simulated_trajectories)

    # get moment expansion result with current parameters
    log_likelihood = 0

    for trajectory in observed_trajectories_lookup.itervalues():
        moment = trajectory.description
        assert(isinstance(moment, Moment))
        assert(moment.order == 1)

        species = np.where(moment.n_vector == 1)[0][0]
        mean_variance = mean_variance_lookup[species]
        if (mean_variance.mean < 0).any() or (mean_variance.variance < 0).any():
            return float('inf')

        term = _eval_density(mean_variance.mean, mean_variance.variance, trajectory.values, distribution)
        log_likelihood += term

    dist = -log_likelihood
    return dist

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


def _eval_density(means, variances,observed_values, distribution):
    """
    Calculates gamma/lognormal/normal pdf given mean variance, x
    where x is the experimental species number measured at a particular timepoint. Returns ln(pdf)
    :param mean: mean
    :param var: variance
    :param observed_values: experimental species number measured at a particular timepoint
    :param distribution: distribution to consider. Either 'gamma', 'normal' or 'lognormal'
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