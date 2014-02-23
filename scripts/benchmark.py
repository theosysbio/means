from means.approximation.mea import MomentExpansionApproximation
from means.examples.sample_models import MODEL_P53
import time



def get_time_for_n_mom(n):
    t0 = time.time()
    pb = MomentExpansionApproximation(MODEL_P53, n, "log-normal").run()
    return (len(pb.right_hand_side),time.time() - t0)

for n in range(1,9):
    print get_time_for_n_mom(n)

#t = log10(c(.5,3,15,66,243)); n_mom = 1:length(t)
#plot(t~n_mom,type = 'b', ylab="log_10(t)")