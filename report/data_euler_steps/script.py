import sys
sys.path.append("../utils")
from report_unit import ReportUnit
from means.examples.sample_models import *
from means.simulation.ssa import SSASimulation
from means.simulation.simulate import Simulation
from means import mea_approximation
from means.core import StochasticProblem
import multiprocessing
import numpy as np


MODEL = MODEL_HES1
RATES = [5, 10, 1, 1]
INITIAL_CONDITIONS = [4, 20, 20]
TMAX = 240
TIME_RANGE = np.arange(0,TMAX,1)
N_SSA = int(2e3)
MAX_ORDER = 7




class MyData(ReportUnit):


    def run(self):

        self.out_object = []
        ssa_means = []

        step=250
        for i in range(0, N_SSA, step):
            print "simulation #{0}".format(i)
            ssas = SSASimulation( StochasticProblem(MODEL), step)
            ssa_means.append(ssas.simulate_system(RATES, INITIAL_CONDITIONS, TIME_RANGE, number_of_processes=8))
        print len(ssa_means)
        mean_trajectories = [sum(trajs)/float(len(trajs)) for trajs in zip(*ssa_means)]

        self.out_object.append( {"method":"SSA", "trajectories": mean_trajectories })
        probl = mea_approximation(MODEL, MAX_ORDER)
        print "mea done. Now simulating ..."
        simulator = Simulation(probl)
        trajects = simulator.simulate_system(RATES, INITIAL_CONDITIONS, TIME_RANGE)

        self.out_object.append( {"method":"MEA_vanilla", "trajectories": trajects})
        try:
            for r in range(3,6):
                h = 1.0 * 2 ** (-r)
                print "Euler Simulation: h step = {0}".format(h)
                simulator = Simulation(probl, solver="euler", h=h)
                trajects = simulator.simulate_system(RATES, INITIAL_CONDITIONS, TIME_RANGE)
                self.out_object.append({"method":"MEA_euler", "step":h, "trajectories": trajects})
        except KeyboardInterrupt:
            pass


MyData()
