"""
This is just a simple script to run test during development
"""
from means.approximation.mea import MomentExpansionApproximation
from means.examples.sample_models import MODEL_P53
from means.approximation.ode_problem import ODEProblemWriter

prob = MomentExpansionApproximation(MODEL_P53, 2, "gamma").run()


#"simple_closure_vs_legacy"

