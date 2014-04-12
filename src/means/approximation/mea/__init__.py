"""
Moment Expansion Approximation
-----

This part of the package implements Moment Expansion Approximation as described in [Ale2013]_.
In addition to the standard implementation, it allows to use different distribution
(such as normal, log-normal and gamma) to close the moment expansion.
The function :func:`mea_approximation` should provide all the necessary options.

    Example:

>>> from means import mea_approximation
>>> from means.examples.sample_models import MODEL_P53
>>> ode_problem = mea_approximation(MODEL_P53,max_order=2)
>>> # equivalent to
>>> # ode_problem = mea_approximation(MODEL_P53, max_order=2, closure="scalar", value=0)
>>> print ode_problem

The result is an :class:`means.core.problems.ODEProblem`. Typically, it would be further used to
perform simulations (see :mod:`~means.simulation`) and inference (see :mod:`~means.inference`).

.. [Ale2013] A. Ale, P. Kirk, and M. P. H. Stumpf,\
   "A general moment expansion method for stochastic kinetic models,"\
   The Journal of Chemical Physics, vol. 138, no. 17, p. 174101, 2013.

------------
"""
from moment_expansion_approximation import MomentExpansionApproximation, mea_approximation