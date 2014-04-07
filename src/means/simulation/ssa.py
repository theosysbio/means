"""
Gillespie Stochastic Simulation Algorithm
----

This part of the package provides a simple implementation of GSSA.
This is designed for experimental purposes much more than for performance.
If you would like to use SSA for parameter inference, or for high
number of species, there are many superior implementations available.
"""

import multiprocessing
import numpy as np
from means.simulation.trajectory import Trajectory, TrajectoryCollection
from means.io.serialise import SerialisableObject
from means.util.moment_counters import generate_n_and_k_counters
from means.util.sympyhelpers import product, substitute_all
from means.core import Moment


class SSASimulation(SerialisableObject):
    """
        A class providing an implementation of the exact Gillespie Stochastic Simulation Algorithm [Gillespie77].

            >>> from means.examples import MODEL_P53
            >>> from means import StochasticProblem, SSASimulation
            >>> import numpy as np
            >>> PROBLEM = StochasticProblem(MODEL_P53)
            >>> RATES = [90, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01]
            >>> INITIAL_CONDITIONS = [70, 30, 60]
            >>> TIME_RANGE = np.arange(0, 40, .1)
            >>> N_SSA = 10
            >>> ssas = SSASimulation(PROBLEM, N_SSA)
            >>> mean_trajectories = ssas.simulate_system(RATES, INITIAL_CONDITIONS, TIME_RANGE)


    .. [Gillespie77]Gillespie, Daniel T. "Exact stochastic simulation of coupled chemical reactions."\
         The journal of physical chemistry 81.25 (1977): 2340-2361.
    """
    def __init__(self, stochastic_problem, n_simulations, random_seed=None):
        """

        :param stochastic_problem:
        :param n_simulations:
        :param random_seed:
        """
        self.__random_seed = random_seed
        self.__problem = stochastic_problem
        self.__n_simulations = n_simulations


    def _validate_parameters(self, parameters, initial_conditions):

        if len(self.__problem.species) != len(initial_conditions):
            exception_str = "The number of initial conditions and the number of species are different. ({0} != {1})"
            raise Exception(exception_str.format(len(self.__problem.species), len(initial_conditions)))

        elif len(self.__problem.parameters) != len(parameters):
            exception_str = "The number of parameters and the number of constants are different. ({0} != {1})"
            raise Exception(exception_str.format(len(self.__problem.parameters), len(parameters)))


    def simulate_system(self, parameters, initial_conditions, timepoints,
                        max_moment_order=1, number_of_processes=1):
        """
        Perform Gillespie SSA simulations and returns trajectories for of each species.
        Each trajectory is interpolated at the given time points.
        By default, the average amounts of species for all simulations is returned.

        :param parameters: list of the initial values for the constants in the model.
                                  Must be in the same order as in the model
        :param initial_conditions: List of the initial values for the equations in the problem.
                        Must be in the same order as these equations occur.

        :param timepoints: A list of time points to simulate the system for

        :param number_of_processes: the number of parallel process to be run
        :param max_moment_order: the highest moment order to calculate the trajectories to.
                                 if set to zero, the individual trajectories will be returned, instead of
                                 the averaged moments.
        E.g. a value of one will return means, a values of two, means, variances and covariance and so on.


        :return: a list of :class:`~means.simulation.Trajectory` one per species in the problem,
            or a list of lists of trajectories (one per simulation) if `return_average == False`.
        :rtype: list[:class:`~means.simulation.Trajectory`]
        """
        max_moment_order = int(max_moment_order)
        assert(max_moment_order >= 0)

        n_simulations = self.__n_simulations
        self._validate_parameters(parameters, initial_conditions)
        t_max= max(timepoints)

        substitution_pairs = dict(zip(self.__problem.parameters, parameters))
        propensities = substitute_all(self.__problem.propensities, substitution_pairs)
        # lambdify the propensities for fast evaluation
        propensities_as_function = self.__problem.propensities_as_function
        def f(*species_parameters):
            return propensities_as_function(*(np.concatenate((species_parameters, parameters))))

        population_rates_as_function = f

        if not self.__random_seed:
            seed_for_processes = [None] * n_simulations
        else:
            seed_for_processes = [i for i in range(self.__random_seed, n_simulations + self.__random_seed)]



        if number_of_processes ==1:
            ssa_generator = _SSAGenerator(population_rates_as_function,
                                        self.__problem.change, self.__problem.species,
                                        initial_conditions, t_max, seed=self.__random_seed)

            results = map(ssa_generator.generate_single_simulation, seed_for_processes)


        else:
            p = multiprocessing.Pool(number_of_processes,
                    initializer=multiprocessing_pool_initialiser,
                    initargs=[population_rates_as_function, self.__problem.change,
                              self.__problem.species,
                              initial_conditions, t_max, self.__random_seed])

            results = p.map(multiprocessing_apply_ssa, seed_for_processes)

            p.close()
            p.join()

        resampled_results = [[traj.resample(timepoints, extrapolate=True) for traj in res] for res in results]
        for i in resampled_results:
            idx = len(i[0].values) - 1

        if max_moment_order == 0:
            # Return a list of TrajectoryCollection objects
            return map(TrajectoryCollection, resampled_results)

        moments = self._compute_moments(resampled_results, max_moment_order)
        return TrajectoryCollection(moments)

    def _compute_moments(self, all_trajectories, max_moment_order):

        mean_trajectories = [sum(trajs)/float(len(trajs)) for trajs in zip(*all_trajectories)]

        if max_moment_order == 1:
            return mean_trajectories
        n_counter, _ = generate_n_and_k_counters(max_moment_order - 1, self.__problem.species)

        out_trajects = mean_trajectories[:]
        for n in n_counter:
            if n.order == 0:
                continue
            out_trajects.append(self._compute_one_moment(all_trajectories, mean_trajectories, n))
        return out_trajects

    def _compute_one_moment(self, all_trajectories, mean_trajectories, moment):

        # the expectation of the product:
        #products_of_sps = [product(trajs) for trajs in all_trajectories]
        n_vec = moment.n_vector

        to_multipl = []

        for i, trajs in enumerate(zip(*all_trajectories)):
            mean_of_sp = mean_trajectories[i]
            order_of_sp = n_vec[i]
            xi_minus_ex = [(t - mean_of_sp) ** order_of_sp for t in trajs]
            for x in xi_minus_ex:
                x.set_description(moment)
            to_multipl.append(xi_minus_ex)

        to_sum = [product(xs) for xs in zip(*to_multipl)]

        return sum(to_sum)/ float(len(to_sum))



def multiprocessing_pool_initialiser(population_rates_as_function, change, species,
                                     initial_conditions, t_max, seed):
    global ssa_generator
    current = multiprocessing.current_process()
    #increment the random seed inside each process at creation, so the result should be reproducible
    if seed:
        seed += current._identity[0]
    else:
        seed = current._identity[0]
    ssa_generator = _SSAGenerator(population_rates_as_function, change, species, initial_conditions, t_max, seed)

def multiprocessing_apply_ssa(x):
    """
    Used in the SSASimulation class.
    Needs to be in global scope for multiprocessing module to pick it up
    """
    result = ssa_generator.generate_single_simulation(x)
    return result


class _SSAGenerator(object):
    def __init__(self, population_rates_as_function, change, species, initial_conditions, t_max, seed):
        """
        :param population_rates_as_function: function to evaluate propensities given the amount of species
        :param change: the change matrix (transpose of the stoichiometry matrix) as an numpy in array
        :param initial_conditions: the initial conditions of the system
        :param t_max: the time when the simulation should stop
        :param seed: an integer to initialise the random seed. If `None`, the random seed will be set
                automatically (e.g. from /dev/random) once for all.
        """
        self.__rng = np.random.RandomState(seed)
        self.__population_rates_as_function = population_rates_as_function
        self.__change = change
        self.__initial_conditions = initial_conditions
        self.__t_max = t_max
        self.__species = species

    def _gssa(self, initial_conditions, t_max):
        """
        This function is inspired from Yoav Ram's code available at:
        http://nbviewer.ipython.org/github/yoavram/ipython-notebooks/blob/master/GSSA.ipynb

        :param initial_conditions: the initial conditions of the system
        :param t_max:  the time when the simulation should stop
        :return:
        """
        # set the initial conditions and t0 = 0.
        species_over_time = [np.array(initial_conditions).astype("int16")]
        t = 0
        time_points = [t]
        while t < t_max and species_over_time[-1].sum() > 0:
            last = species_over_time[-1]
            e, dt = self._draw(last)
            t += dt
            species_over_time.append(last + self.__change[e,:])
            time_points.append(t)
        return time_points, np.array(species_over_time).T

    def _draw(self, population):
        population_rates = np.array(self.__population_rates_as_function(*population))
        sum_rates = population_rates.sum()
        time = self.__rng.exponential(1.0/sum_rates)
        event_distribution = population_rates.flatten()
        event_distribution /= event_distribution.sum()
        event = self.__rng.multinomial(1, event_distribution).argmax()
        return event, time

    def generate_single_simulation(self, x):
        """
        Generate a single SSA simulation
        :param x: an integer to reset the random seed. If None, the initial random number generator is used
        :return: a list of :class:`~means.simulation.Trajectory` one per species in the problem
        :rtype: list[:class:`~means.simulation.Trajectory`]
        """
        #reset random seed
        if x:
            self.__rng = np.random.RandomState(x)

        # perform one stochastic simulation
        time_points, species_over_time = self._gssa(self.__initial_conditions, self.__t_max)

        # build descriptors for first order raw moments aka expectations (e.g. [1, 0, 0], [0, 1, 0] and [0, 0, 1])
        descriptors = []
        for i, s in enumerate(self.__species):
            row = [0] * len(self.__species)
            row[i] = 1
            descriptors.append(Moment(row, s))

        # build trajectories
        trajectories = [Trajectory(time_points, spot, desc) for
                        spot, desc in zip(species_over_time, descriptors)]

        return trajectories



