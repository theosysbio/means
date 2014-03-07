# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#%pylab inline

# <codecell>

from means.approximation.mea.moment_expansion_approximation import mea_approximation
import means
from means.examples.sample_models import *

# <codecell>

import numpy as np
import sympy

# <codecell>

model = MODEL_P53
constant_values = [90, 0.002, 1.7, 1.1, 0.93, 0.96, 0.01] 
initial_conditions = [70, 30, 60]
timepoints = np.arange(0, 40, .1)

# <codecell>

closer_args = [
               {"closer":"zero"},
               {"closer":"log-normal", "multivariate":True},
               {"closer":"log-normal", "multivariate":False},
               {"closer":"normal", "multivariate":True},
               {"closer":"normal", "multivariate":False}
               ]
        
        

# <codecell>

import pickle
    
all_results = []
try:
    for max_order in range(2,11):
        for cl_arg in closer_args:
            print "simulating for max_order = {0} and closer = {1}".format(max_order, cl_arg)
            probl = mea_approximation(MODEL_P53, max_order, **cl_arg)
            simulator = means.simulation.Simulation(probl, solver='cvode', discr="BDF",maxord=5, maxh=0.01)
            try:
                trajects = simulator.simulate_system(constant_values, initial_conditions, timepoints)
            except Exception as e:
                print "FAILED WITH {0!r}".format(e)
                trajects = None
                pass
    
            
            if trajects:
                result = dict(cl_arg)
                result["max_order"] = max_order
                result["trajectories"] = trajects
                
                all_results.append(result)
                if  cl_arg.has_key("multivariate") and cl_arg["multivariate"]:
                    multi_or_uni_variate =  "multivariate"
                else:
                    multi_or_uni_variate = 'univariate'
                result["multi_or_uni_variate"] = multi_or_uni_variate
                file_name = "traject_{0}_{1}_{2}.pickle".format(max_order, cl_arg["closer"], multi_or_uni_variate)
                #here I will save result under `file_name`
                
                with open(file_name, 'w') as f:
                    pickle.dump(result, f)
finally:
    with open("traject_everything.pickle", 'w') as f:
        pickle.dump(all_results, f)

# <codecell>


from matplotlib import pyplot as plt

# <codecell>

for n in range(3):
    plt.figure()
    for ar in all_results:
        if ar["trajectories"] :
            ar["trajectories"][n].plot(label = "{0}_{1}_{2}".format(ar["max_order"], ar["closer"], ar["multi_or_uni_variate"]))

            plt.legend(loc='lower left', bbox_to_anchor = (1, 0))

    plt.title("Moment : {0}".format(ar["trajectories"][n].description))
    plt.show()
plt.close() 

# <codecell>


