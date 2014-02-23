
from means.approximation.mea import MomentExpansionApproximation
from means.examples.sample_models import MODEL_P53
import time
import sys


def get_time_for_n_mom(n):
    t0 = time.time()
    pb = MomentExpansionApproximation(MODEL_P53, n, "log-normal").run()
    return (len(pb.right_hand_side), (time.time() - t0))

MAX_ORDER = int(sys.argv[2])
tag = sys.argv[1]
res = [get_time_for_n_mom(n) for n in range(2, MAX_ORDER+1)]
for ne,t in res:
    print "{0}, {1}, {2}".format(tag, ne, t)
