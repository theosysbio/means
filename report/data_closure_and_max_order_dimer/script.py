import sys
sys.path.append("../utils")
from report_unit import ReportUnit
from means.examples.sample_models import *
from means.simulation.ssa import SSASimulation
from means.simulation.simulate import Simulation
from means.approximation.mea.moment_expansion_approximation import mea_approximation
from means.core import StochasticProblem
import multiprocessing
import numpy as np


MODEL = MODEL_DIMERISATION

RATES = [0.001,	0.5, 330]
INITIAL_CONDITIONS = [320]
TMAX = 30
TIME_RANGE = np.arange(0,TMAX,.1)
N_SSA = int(1e4)
MAX_ORDER = 8


def get_one_mea_result(max_order_cl_arg):
    try:
        max_order, cl_arg = max_order_cl_arg

        print (max_order, cl_arg)
        probl = mea_approximation(MODEL, max_order, **cl_arg)
        simulator = Simulation(probl)

        try:
            trajects = simulator.simulate_system(RATES, INITIAL_CONDITIONS, TIME_RANGE)
        except Exception as e:
            print "FAILED WITH {0!r}".format(e)
            return

        result = dict(cl_arg)
        result["method"] = "MEA"
        result["max_order"] = max_order
        #keep only the means
        result["trajectories"] = trajects[0:MODEL.number_of_species]
        return result
    except KeyboardInterrupt:
        return

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
        print ssa_means
        mean_trajectories = [sum(trajs)/float(len(trajs)) for trajs in zip(*ssa_means)]

        self.out_object.append( {"method":"SSA", "trajectories": mean_trajectories })

        closer_args = [
                       {"closure":"scalar"},
                       {"closure":"log-normal", "multivariate":False},
                       {"closure":"normal", "multivariate":False}
                       ]

        try:
            for max_order in range(2,MAX_ORDER+1):

                pool = multiprocessing.Pool(processes=len(closer_args))
                result_dicts= pool.map(get_one_mea_result,zip([max_order]*len(closer_args),closer_args) )
                pool.close()
                pool.join()
                for result in result_dicts:
                    if result:
                        self.out_object.append(result)

        except KeyboardInterrupt:
            return
            pass


MyData()
