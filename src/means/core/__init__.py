"""
This package defines the common classes that are used within all of the other means subpackages.

The module exposes the descriptor classes, such as :class:`~means.descriptors.Descriptor`,
:class:`~means.descriptors.VarianceTerm`, :class:`~means.descriptors.Moment` and
:class:`~means.descriptors.ODETermBase` that are used to describe the types of trajectories generated,
as well as certain terms in the ODE equations.

Similarly, both the :class:`~means.problem.StochasticProblem` and
:class:`~means.problem.ODEProblem` classes that are used in stochastic and deterministic simulations respectively
are exposed by this module.

Finally, the :class:`Model` class, which provides a standard interface to describe a biological model,
and can be thought to be the center of the whole package, is also implemented here.
"""
from .descriptors import Descriptor, VarianceTerm, Moment, ODETermBase
from .problems import StochasticProblem, ODEProblem
from .model import Model
