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
INITIAL_CONDITIONS = [70, 30, 60]
TMAX = 40
TIME_RANGE = np.arange(0,TMAX,.1)
N_SSA = 10
MAX_ORDER = 9



# we run 10 simulations
#trajectories = map(get_one_ssa_traj, [i for i in range(10)])
# and calculate the means per species


ssa_simulator = SSASimulator(MODEL)
def get_one_ssa_traj(i):
    print i
    # here the trick it to set the random seed according to i so that
    # we have different results for different processes
    ssa_simulator.reset_random_seed(i)
    one_run_trajectories = ssa_simulator.simulate_system(RATES, INITIAL_CONDITIONS, TMAX)
    one_run_trajectories = [tr.resample(TIME_RANGE) for tr in one_run_trajectories]
    return one_run_trajectories

def get_one_mea_result(max_order_cl_arg):
    try:
        max_order, cl_arg = max_order_cl_arg

        print (max_order, cl_arg)
        probl = run_mea(MODEL, max_order, **cl_arg)
        simulator = Simulation(probl, solver='cvode', discr="BDF",maxord=5, maxh=0.01, maxsteps=1000)

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


    def get_all_sss_means(self):

        pool = multiprocessing.Pool(processes=7)
        trajectories = pool.map(get_one_ssa_traj, [i for i in range(N_SSA)])

        pool.close()
        pool.join()
        #trajectories = map(self.get_one_ssa_traj, [i for i in range(N_SSA)])

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
        print "?"
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
