from __future__ import absolute_import, print_function
"""
Linear Noise Approximation
-----

This part of the package implements Linear Noise Approximation as described in [Komorowski2009]_.

Example:

>>> from means.approximation.lna.lna import lna_approximation
>>> from means.examples.sample_models import MODEL_P53
>>> ode_problem = lna_approximation(MODEL_P53)
>>> print(ode_problem)

The result is an :class:`means.core.problems.ODEProblem`. Typically, it would be further used to
perform simulations (see :mod:`~means.simulation`) and inference (see :mod:`~means.inference`).

.. [Komorowski2009] M. Komorowski, B. Finkenstadt, C. V. Harper, and D. A. Rand,\
"Bayesian inference of biochemical kinetic parameters using the linear noise approximation,"\
BMC Bioinformatics, vol. 10, no. 1, p. 343, Oct. 2009.

------------
"""
from .lna import LinearNoiseApproximation, lna_approximation
