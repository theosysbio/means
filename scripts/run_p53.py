"""
This is just a simple script to run test during development
"""
from means.approximation.mea.moment_expansion_approximation import run_mea

from means.examples.sample_models import MODEL_P53

prob = run_mea(MODEL_P53, 4, "log-normal")

#"simple_closure_vs_legacy"
