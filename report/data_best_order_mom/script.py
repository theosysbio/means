import sys
sys.path.append("../utils")
from report_unit import ReportUnit
from means.examples.sample_models import *
from means.simulation.ssa import SSASimulator
from means.simulation.simulate import Simulation
from means.approximation.mea.moment_expansion_approximation import run_mea

import multiprocessing
import numpy as np


MODEL = MODEL_P53

RATES = [90, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01]
INITIAL_CONDITIONS = [10, 20, 30]
TMAX = 40
TIME_RANGE = np.arange(0,TMAX,.1)
N_SSA = 200
MAX_ORDER = 5



# we run 10 simulations
#trajectories = map(get_one_ssa_traj, [i for i in range(10)])
# and calculate the means per species



# <codecell>

class MyData(ReportUnit):

    def get_one_ssa_traj(self, i):
        print i
        # here the trick it to set the random seed according to i so that
        # we have different results for different processes
        self.ssa_simulator.reset_random_seed(i)
        one_run_trajectories = self.ssa_simulator.simulate_system(RATES, INITIAL_CONDITIONS, TMAX)
        one_run_trajectories = [tr.resample(TIME_RANGE) for tr in one_run_trajectories]
        return one_run_trajectories

    def get_all_sss_means(self):
        self.ssa_simulator = SSASimulator(MODEL)
        # pool = multiprocessing.Pool(processes=4)
        # trajectories = pool.map(self.get_one_ssa_traj, [i for i in range(N_SSA)])
        # #
        # pool.close()
        # pool.join()
        trajectories = map(self.get_one_ssa_traj, [i for i in range(N_SSA)])

        mean_trajectories = [sum(trajs)/len(trajs) for trajs in zip(*trajectories)]
        return mean_trajectories

    def run(self):

        self.out_object = []
        ssa_means = self.get_all_sss_means()
        self.out_object.append( {"method":"SSA", "trajectories": ssa_means})

        closer_args = [
                       {"closer":"zero"},
                       {"closer":"log-normal", "multivariate":True},
                       {"closer":"log-normal", "multivariate":False},
                       {"closer":"normal", "multivariate":True},
                       {"closer":"normal", "multivariate":False}
                       ]


        for max_order in range(2,MAX_ORDER+1):
            for cl_arg in closer_args:
                print (max_order, cl_arg)
                probl = run_mea(MODEL_P53, max_order, **cl_arg)
                simulator = Simulation(probl, solver='cvode', discr="BDF",maxord=5, maxh=0.01)

                try:
                    trajects = simulator.simulate_system(RATES, INITIAL_CONDITIONS, TIME_RANGE)
                    #invalid trajects:
                    # if max([max(t.values) for t in trajects]) > REJECTED_TRAJ_THRESHOLD:
                    #     raise Exception("rejected trajectory")
                except Exception as e:
                    print "FAILED WITH {0!r}".format(e)
                    trajects = None
                    pass


                if trajects:
                    result = dict(cl_arg)
                    result["method"] = "MEA"
                    result["max_order"] = max_order
                    #keep only the means
                    result["trajectories"] = trajects[0:MODEL.number_of_species]

                    self.out_object.append(result)


        


MyData()
