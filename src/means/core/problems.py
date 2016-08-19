from __future__ import absolute_import, print_function
"""
Problems
--------

This part of the package implement classes describing "problems".
Problems are required inputs for simulation and inference.
Currently, there are two types of problems:

* A :class:`ODEProblem` is a system of differential equations describing
    the temporal behaviour of the system. They are typically obtained through approximation (e.g. \
    :mod:`~means.approximation.mea.moment_expansion_approximation`, or \
    :mod:`~means.approximation.lna.lna`) \

* A :class:`StochasticProblem` can be used for stochastic simulations\
and can be simply built from  a :class:`~means.core.model.Model`:

>>> from means import StochasticProblem
>>> from means.examples.sample_models import MODEL_P53
>>> my_stoch_prob = StochasticProblem(MODEL_P53)

"""


import sympy
import numpy as np
from sympy.utilities.autowrap import autowrap

from means.core.model import Model
from means.core.descriptors import Moment
from means.compat import string_types, unicode_compatible, iteritems
from means.io.latex import LatexPrintableObject
from means.io.serialise import SerialisableObject
from means.util.memoisation import memoised_property, MemoisableObject
from means.util.sympyhelpers import to_list_of_symbols, to_sympy_column_matrix, to_sympy_matrix
from means.util.sympyhelpers import sympy_expressions_equal


@unicode_compatible
class ODEProblem(SerialisableObject, LatexPrintableObject, MemoisableObject):
    """
    Creates a `ODEProblem` object that stores a system of ODEs describing the kinetic of a system.
    Typically, `ODEProblem`s will be further used in simulations (see :mod:`~means.simulation`)
    and inference (see :mod:`~means.inference`).

    """

    # These are private (as indicated by __, the code is a bit messier, but we can ensure immutability this way)
    __right_hand_side = None
    __left_hand_side = None
    __parameters = None

    yaml_tag = '!problem'

    def __init__(self, method, left_hand_side_descriptors, right_hand_side, parameters):
        """
        :param method: a string describing the method used to generate the problem.
        Currently, 'MEA' and 'LNA' are supported"
        :param left_hand_side_descriptors: the left hand side of equations as a list of
            :class:`~means.core.descriptors.Descriptor` objects (such as :class:`~means.core.descriptors.Moment`)
        :param right_hand_side: the right hand side of equations
        :param parameters: the parameters of the model
        """

        self.__left_hand_side_descriptors = left_hand_side_descriptors
        self.__left_hand_side = to_sympy_column_matrix(to_sympy_matrix(
                    [plhs.symbol for plhs in left_hand_side_descriptors])
                    )
        self.__right_hand_side = to_sympy_column_matrix(right_hand_side)
        self.__parameters = to_list_of_symbols(parameters)
        self.__method = method

    def validate(self):
        """
        Validates whether the ODE equations provided make sense  i.e. the number of right-hand side equations
        match the number of left-hand side equations.
        """
        if self.left_hand_side.rows != self.right_hand_side.rows:
            raise ValueError("There are {0} left hand side equations and {1} right hand side equations. "
                             "The same number is expected.".format(self.left_hand_side.rows, self.right_hand_side.rows))

    # Expose public interface for the specified instance variables
    # Note that all properties here are "getters" only, thus assignment won't work
    @property
    def left_hand_side_descriptors(self):
        return self.__left_hand_side_descriptors

    @property
    def left_hand_side(self):
        return self.__left_hand_side

    @property
    def variables(self):
        return to_list_of_symbols(self.__left_hand_side)

    @property
    def number_of_species(self):
        species = [it[1]  for it in iteritems(self._descriptions_dict) if
            isinstance(it[1], Moment) and it[1].order == 1]

        return len(species)

    @property
    def right_hand_side(self):
        return self.__right_hand_side

    @property
    def parameters(self):
        return self.__parameters

    @property
    def number_of_parameters(self):
        return len(self.parameters)

    @property
    def method(self):
        return self.__method

    @memoised_property
    def _descriptions_dict(self):
        return {ode_term.symbol: ode_term for ode_term in self.left_hand_side_descriptors}

    @property
    def number_of_equations(self):
        return len(self.left_hand_side)

    @memoised_property
    def _right_hand_side_as_numeric_functions(self):
        all_symbols = self.parameters + self.variables
        wrapping_func = lambda x: autowrap(x, args=all_symbols, language='C', backend='Cython')
        return map(wrapping_func, self.right_hand_side)

    @memoised_property
    def right_hand_side_as_function(self):
        """
        Generates and returns the right hand side of the model as a callable function that takes two parameters:
        values for variables and values for constants,
        e.g. `f(values_for_variables=[1,2,3], values_for_constants=[3,4,5])

        This function is directly used in `means.simulation.Simulation`
        :return:
        :rtype: function
        """
        wrapped_functions = self._right_hand_side_as_numeric_functions

        def f(values_for_variables, values_for_constants):
            all_values = np.concatenate((values_for_constants, values_for_variables))
            ans = np.array([w_f(*all_values) for w_f in wrapped_functions])
            return ans

        return f

    def descriptor_for_symbol(self, symbol):
        """
        Given the symbol associated with the problem.
        Returns the :class:`~means.core.descriptors.Descriptor` associated with that symbol

        :param symbol: Symbol
        :type symbol: basestring|:class:`sympy.Symbol`
        :return:
        """
        if isinstance(symbol, string_types):
            symbol = sympy.Symbol(symbol)

        try:
            return self._descriptions_dict[symbol]
        except KeyError:
            raise KeyError("Symbol {0!r} not found in left-hand-side of the equations".format(symbol))

    def __str__(self):
        equations_pretty_str = '\n\n'.join(['{0!r}:\n    {1!r}'.format(x, y) for x, y in zip(self.left_hand_side_descriptors,
                                                                                           self.right_hand_side)])
        return u"{0.__class__!r}\n" \
               u"Method: {0.method!r}\n" \
               u"Parameters: {0.parameters!r}\n" \
               u"\n" \
               u"Equations:\n\n" \
               u"{1}\n".format(self, equations_pretty_str)

    def __repr__(self):
        return str(self)

    def _repr_latex_(self):
        """
        This is used in IPython notebook it allows us to render the ODEProblem object in LaTeX.
        How Cool is this?
        """
        # TODO: we're mixing HTML with latex here. That is not necessarily a good idea, but works
        # with IPython 1.2.0. Once IPython 2.0 is released, this needs to be changed to _ipython_display_
        lines = []
        lines.append(r"<h1>{0}</h1>".format(self.__class__.__name__))

        lines.append("<p>Method: <code>{0!r}</code></p>".format(self.method))
        lines.append("<p>Parameters: <code>{0!r}</code></p>".format(self.parameters))
        lines.append("<p>Terms:</p>")
        lines.append("<ul>")
        lines.extend(['<li><code>{0!r}</code></li>'.format(lhs) for lhs in self.left_hand_side_descriptors])
        lines.append("</ul>")
        lines.append('<hr />')
        lines.append(r"\begin{align*}")
        for lhs, rhs in zip(self.left_hand_side_descriptors, self.right_hand_side):
            lines.append(r"\dot{{{0}}} &= {1} \\".format(sympy.latex(lhs.symbol), sympy.latex(rhs)))
        lines.append(r"\end{align*}")
        return "\n".join(lines)

    @property
    def latex(self):
        STRING_RIGHT_HAND = 'RHS of equations:'
        STRING_MOM = 'List of moments:'

        left_hand_side = self.left_hand_side_descriptors
        preamble = ["\\documentclass{article}"]
        preamble += ["\\usepackage[landscape, margin=0.5in, a3paper]{geometry}"]
        lines = ["\\begin{document}"]
        lines += ["\\section*{%s}" % STRING_RIGHT_HAND]

        lines += ["$\\dot {0} = {1} {2}$".format(str(sympy.latex(lhs.symbol)), str(sympy.latex(rhs)), r"\\")
                    for (rhs, lhs) in zip(self.right_hand_side, left_hand_side)]

        lines += [r"\\"] * 5

        lines += ["\\section*{%s}" % STRING_MOM]


        lines += ["$\\dot {0}$: {1} {2}".format(str(sympy.latex(lhs.symbol)), str(lhs), r"\\")
                       for lhs in left_hand_side if isinstance(lhs, Moment)]

        lines += ["\\end{document}"]

        return '\n'.join(preamble + lines)


    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.parameters == other.parameters \
                   and other.left_hand_side_descriptors == self.left_hand_side_descriptors \
                   and sympy_expressions_equal(other.right_hand_side, self.right_hand_side)

    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = [('method', data.method),
                   ('parameters', map(str, data.parameters)),
                   ('left_hand_side_descriptors', list(data.left_hand_side_descriptors)),
                   ('right_hand_side', map(str, data.right_hand_side))]

        return dumper.represent_mapping(cls.yaml_tag, mapping)



class StochasticProblem(Model, MemoisableObject):
    """
    The formulation of a model for stochastic simulations such as GSSA (see :mod:`means.simulation.ssa`).
    """
    def __init__(self, model):
        super(StochasticProblem, self).__init__(model.species, model.parameters,
                                                model.propensities, model.stoichiometry_matrix)
        self.__change = np.array(model.stoichiometry_matrix.T).astype("int")

    @property
    def change(self):
        return self.__change

    @memoised_property
    def propensities_as_function(self):
        all_symbols = self.species + self.parameters
        wrapping_func = lambda x: autowrap(x, args=all_symbols, language='C', backend='Cython')
        wrapped_functions = map(wrapping_func, self.propensities)

        def f(*args):
            ans = np.array([w_f(*args) for w_f in wrapped_functions])
            return ans

        return f
