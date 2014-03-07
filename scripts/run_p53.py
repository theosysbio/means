"""
This is just a simple script to run test during development
"""
from means.approximation.mea.moment_expansion_approximation import mea_approximation

from means.examples.sample_models import MODEL_P53

prob = mea_approximation(MODEL_P53, 4, "log-normal", multivariate=True)

#"simple_closure_vs_legacy"
