from means.simulation.trajectory import Trajectory
from means.io.serialise import SerialisableObject
import numpy as np
import sympy as sp

from means.util.sympyhelpers import substitute_all, to_sympy_matrix


class SSASimulator(SerialisableObject):
    """
    An implementation of the exact Gillespie Stochastic Simulation Algorithm [Gillespie77].


    .. [Gillespie77]Gillespie, Daniel T. "Exact stochastic simulation of coupled chemical reactions."\
         The journal of physical chemistry 81.25 (1977): 2340-2361.
    """
    def __init__(self, model, random_seed=None):
        self.__rng = np.random.RandomState(random_seed)
        self.__model = model
        self.__species = to_sympy_matrix(self.__model.species)

    def reset_random_seed(self, random_seed=None):
        self.__rng = np.random.RandomState(random_seed)

    def simulate_system(self, parameters, initial_conditions, t_max):

        """
        Performs one SSA simulation for the given parameters and initial conditions.

        :param parameters: list of the initial values for the constants in the model.
                                  Must be in the same order as in the model.

        :param initial_conditions: List of the initial values for the equations in the problem.
                        Must be in the same order as these equations occur.
                               If not all values specified, the remaining ones will be assumed to be 0.

        :param t_max: The time when the simulation should stop. The simulation will
         stop before if no species are present (since, in this case, no reaction can occur)

        :return:
        """
        self.change = np.array(self.__model.stoichiometry_matrix.T).astype("int")

        # evaluate substitute rates by their actual values
        substitution_pairs = dict(zip(self.__model.constants, parameters))
        propensities =  substitute_all(self.__model.propensities, substitution_pairs)

        # lambdify the propensities for fast evaluation
        self.population_rates_as_function = sp.lambdify(tuple(self.__species),
                                                        propensities, modules="numpy")

        # perform one stochastic simulation
        time_points, species_over_time = self._gssa(initial_conditions,t_max)

        # build trajectories
        trajectories = [Trajectory(time_points,spot,desc) for
                        spot,desc in zip(species_over_time,self.__model.species)]

        return trajectories

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
        #    print last
            e, dt = self._draw(last)
            t += dt
            species_over_time.append(last + self.change[e,:])
            time_points.append(t)
        return time_points, np.array(species_over_time).T

    def _draw(self, population):
        population_rates = np.array(self.population_rates_as_function(*population))
        sum_rates = population_rates.sum()
        time = self.__rng.exponential(1.0/sum_rates)
        event_distribution = population_rates.flatten()
        event_distribution /= event_distribution.sum()
        event = self.__rng.multinomial(1, event_distribution).argmax()
        return event, time


