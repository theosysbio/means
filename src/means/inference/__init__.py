"""
Parameter Inference
-----

This part of the package provides utilities for parameter inference.
Parameter inference will try to find the set of parameters which
produces trajectories with minimal distance to the observed trajectories.
Different distance functions are implemented (such as functions minimising the sum of squares error
or functions based on parametric likelihood), but it is also possible to use custom distance functions.

The package provides support for both inference from a single starting point (:class:`~means.inference.Inference`)
or inference from random starting points (:class:`~means.inference.InferenceWithRestarts`).

Some basic inference result plotting functionality is also provided by the package, see the documentation for
:class:`~means.inference.InferenceResult` for more information on this.
"""
from inference import *
from results import InferenceResult, InferenceResultsCollection
