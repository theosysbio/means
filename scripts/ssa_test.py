from means.core.problem import StochasticProblem
from means.simulation.ssa import SSASimulation
import numpy as np
from means.examples.sample_models import *


>>> MODEL = MODEL_P53
>>> RATES = [90, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01]
>>> INITIAL_CONDITIONS = [70, 30, 60]
>>> TIME_RANGE = np.arange(0, 40, .1)
>>> N_SSA = 16
>>> problem = StochasticProblem(MODEL)
>>> ssas = SSASimulation(problem,  random_seed=None)
>>> mean_trajectories = ssas.simulate_system(RATES, INITIAL_CONDITIONS, TIME_RANGE, N_SSA, number_of_processes=4)


import pylab as pl
pl.figure()
for tr in mean_trajectories:
    tr.plot()
pl.show()