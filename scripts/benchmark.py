from means.approximation.mea import MomentExpansionApproximation
from means.examples.sample_models import MODEL_P53
import time



def get_time_for_n_mom(n):
    t0 = time.time()
    MomentExpansionApproximation(MODEL_P53, n, "normal").run()
    return time.time() - t0

for n in range(2,9):
    print (n,get_time_for_n_mom(n))