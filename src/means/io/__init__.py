from __future__ import absolute_import, print_function
"""
Module for Input/Output operations.
-----------------------------------

Human Readable Serialisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module implements common methods to serialise and deserialise MEANS objects.
Namely the module provides functions :func:`means.io.dump` and :func:`means.io.load` that would
serialise and deserialise the said objects into :mod:`yaml` format.
These serialised representations can be written to or read from files with the help of
:func:`means.io.to_file` and :func:`means.io.from_file` functions.

For the user's convenience, the said methods are also attached to all serialisable objects,
e.g. :meth:`means.core.Model.from_file()` method would allow the user
to read :class:`means.core.Model` object from file directly.

Binary Serialisation
~~~~~~~~~~~~~~~~~~~~
We do not provide any convenience functions for binary serialisation of the object, because :mod:`pickle` package,
which is in the default distribution of Python, has no problems of performing these tasks on MEANS objects.

We recommend using :mod:`pickle`, rather than :mod:`means.io` whenever fast serialisation is preferred to human
readability.

SBML
~~~~
This module also provides support for the input from SBML files.
If the :mod:`libsbml` is installed in the user's system and has the appropriate python bindings,
the function :func:`means.io.read_sbml` can be used to parse the files in SBML format
to :class:`means.core.Model` objects.
"""
from .serialise import dump, load, to_file, from_file
from .sbml import read_sbml
