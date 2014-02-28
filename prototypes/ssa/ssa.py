from means.examples.sample_models import *
from means.simulation.trajectory import Trajectory
from numpy.random import exponential, multinomial, gamma
import numpy as np
import sympy as sp

from means.util.sympyhelpers import substitute_all, to_sympy_matrix


class SSASimulator(object):
    def __init__(self, model, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        self.__model = model

        self.species = self.__model.species
    def simulate(self,tmax):



        self.change = np.array(self.__model.stoichiometry_matrix.T).astype("int")

        self.rates = [90, 0.002, 1.7, 1.2, 0.93, 0.96, 0.01]
        initial_conditions = [10, 20, 30]

        subtitution_pairs = dict(zip(model.constants,self.rates))
        evaluated_propensities =  substitute_all(model.propensities, subtitution_pairs)

        self.population_rates_as_function = sp.lambdify(tuple(to_sympy_matrix(self.__model.species)),
                                                        evaluated_propensities, modules="numpy")


        t, ys = self.gssa(initial_conditions, tmax)

        trajectories = [Trajectory(t,y,spe) for y,spe in zip(ys,self.__model.species)]


        return trajectories


    def draw(self, population):

        population_rates = np.array(self.population_rates_as_function(*population))

        sum_rates = population_rates.sum()

        time = exponential(1.0/sum_rates)
        event_distribution = population_rates.flatten()
        event_distribution /= event_distribution.sum()
        event = multinomial(1, event_distribution).argmax()
        return event,time

    def gssa(self, y0, t_max):
        y = [np.array(y0).astype("int")]
        t = 0
        i = [t]

        #self.rates[3] = gamma(12,0.1)


        while t < t_max and np.sum(y[-1]) > 0:


            e,t_ = self.draw(y[-1])
            t += t_
            y.append(y[-1] + self.change[e,:])
            i.append(t)

        y = np.array(y)
        i = np.array(i)

        ys = [j for j in y.T]

        return i, ys

model = MODEL_P53

trajectories = []

ssa_simulator = SSASimulator(model)

def get_one_ssa_traj(_):
    print _
    one_run_trajectories = ssa_simulator.simulate(40)
    one_run_trajectories = [tr.resample(np.arange(0,40,.5)) for tr in one_run_trajectories]
    return one_run_trajectories

import multiprocessing
pool = multiprocessing.Pool(processes=4)
trajectories = pool.map(get_one_ssa_traj, [_ for _ in range(100)])

pool.close()
pool.join()

mean_trajectories = [sum(trajs)/len(trajs) for trajs in zip(*trajectories)]








# yy = reduce(operator.add, y) / N
import pylab as pl
pl.figure()

for tr in mean_trajectories:
    tr.plot(linewidth=2.5)

pl.legend()

# pl.plot(yy)
#
pl.show()

