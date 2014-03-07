import sympy
import numpy as np
from sympy.utilities.autowrap import autowrap
from means.io.latex import LatexPrintableObject
from means.io.serialise import SerialisableObject
from means.util.memoisation import memoised_property, MemoisableObject
from means.util.sympyhelpers import to_list_of_symbols, to_sympy_column_matrix, to_sympy_matrix, to_one_dim_array
from means.util.sympyhelpers import sympy_expressions_equal

class Descriptor(SerialisableObject):
    yaml_tag = u"!descriptor"

class ODETermBase(Descriptor):
    """
    Base class for explaining terms in the ODE expressions.
    Instances of this class allow providing a description for each of the equations in the generated ODE system.
    """

    _symbol = None

    def __init__(self, symbol):
        super(ODETermBase, self).__init__()

        # Sometimes we want to code the moment as sympy.Integer(1) for instance to reduce number of calculations
        if isinstance(symbol, int):
            symbol = sympy.Integer(symbol)

        if symbol is not None and not isinstance(symbol, sympy.Symbol) and not isinstance(symbol, sympy.Integer):
            symbol = sympy.Symbol(symbol)

        self._symbol = symbol

    @property
    def symbol(self):
        return self._symbol

    @property
    def descriptor(self):
        """
        Returns an uniquely identifying descriptor for this particular ODE term.
        """
        return None

    def __repr__(self):
        return str(self)

    def __str__(self):
        return unicode(self).encode('utf8')

    def __unicode__(self):
        return u'{0}({1})'.format(self.__class__.__name__, self.symbol)

    def __mathtext__(self):
        # Double {{ and }} in multiple places as to escape the curly braces in \frac{} from .format
        return r'${0}$'.format(self.symbol)

    def _repr_latex(self):
        return '${0}$'.format(self.symbol)



class VarianceTerm(ODETermBase):
    """
    Signifies that a particular equation generated from the model is part of a Variance Term
    """
    _position = None

    yaml_tag = '!variance-term'

    def __init__(self, position, symbol):
        """
        Creates a Descriptor for a particular ODE in the system that signifies that that particular equation
        computes the position-th term of a covariance matrix, where position is some tuple (row,column).

        It is used in LNA approximation as there we need to deal with moment and variance terms differently

        :param position: position in the covariance matrix
        :param symbol: symbol assigned to the term
        """
        super(VarianceTerm, self).__init__(symbol=symbol)
        self._position = position

    @property
    def position(self):
        return self._position

    def __unicode__(self):
        return u'{0}({1}, {2})'.format(self.__class__.__name__, self.symbol, self.position)

    def _repr_latex_(self):
        return '${0}$ (Variance term $V_{{{0}, {1}}})'.format(self.symbol, self.position[0], self.position[1])

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.symbol == other.symbol and self.position == other.position

    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = [('symbol', str(data.symbol)), ('position', data.position)]
        return dumper.represent_mapping(cls.yaml_tag, mapping)


class Moment(ODETermBase):
    """
    An annotator for ODE expressions that describes that a particular expression in a set of ODEs corresponds to a Moment
    of the probability distribution. The particular moment is described by :attr:`Moment.n_vector`.
    """
    __n_vector = None

    yaml_tag = u'!moment'

    def __init__(self, n_vector, symbol):
        """
        Creates an ODETerm that describes that a particular ODE term is a moment defined by the `n_vector`.
        Should be a vector of ints.

        TODO: figure out what "n_vector" is in mathematics-speak and use this here
        :param n_vector: a vector specifying the multidimensional moment
        """
        super(Moment, self).__init__(symbol=symbol)

        self.__n_vector = np.array(n_vector, dtype=int)
        self.__order = sum(self.n_vector)
        self.__descriptor = self.n_vector

    @property
    def descriptor(self):
        return self.__n_vector

    @property
    def n_vector(self):
        """
        The n_vector this moment represents
        """
        return self.__n_vector

    @property
    def order(self):
        """
        The order of the moment
        """
        return self.__order

    @property
    def is_mixed(self):
        """
        Returns whether the moment is a mixed moment, i.e. has a non-zero power to more than one species,
        or a raw moment (non-zero power to only one species).
        """
        # If moment is not mixed, it will be of form [0, ... , k, ..., 0] where k is the max order
        return self.order not in self.n_vector

    def __str__(self):
        return ', '.join(map(str, self.n_vector))

    def __hash__(self):
        # Allows moment objects to be stored as keys to dictionaries
        return hash(repr(self.n_vector))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return np.equal(self.n_vector, other.n_vector).all() and self.symbol == other.symbol

    def __ge__(self, other):
        """
        A moment is said greater or equal than another iff all the values of n_vec are greater or equal.
        Mathematically: ::math::`n_i^a \ge n_i^b ~ \textrm{for all i}`
        """
        return (self.n_vector >= other.n_vector).all()

    def __repr__(self):
        return '{0}({1!r}, symbol={2!r})'.format(self.__class__.__name__, self.n_vector, self.symbol)


    def _repr_latex_(self):
        return '{0}($[{1}]$, symbol=${2}$)'.format(self.__class__.__name__, ', '.join(map(str, self.n_vector)), self.symbol)

    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = [('symbol', str(data.symbol)), ('n_vector', data.n_vector.tolist())]
        return dumper.represent_mapping(cls.yaml_tag, mapping)

class ODEProblem(SerialisableObject, LatexPrintableObject, MemoisableObject):
    """
    Stores the left and right hand side equations to be simulated

    """

    # These are private (as indicated by __, the code is a bit messier, but we can ensure immutability this way)
    __right_hand_side = None
    __left_hand_side = None
    __constants = None

    yaml_tag = '!problem'

    def __init__(self, method, left_hand_side_descriptors, right_hand_side, constants):
        """
        Creates a `ODEProblem` object that stores the problem to be simulated/used for inference
        :param method: a string describing the method used to generate the problem.
        Currently, 'MEA' and 'LNA' are supported"
        :param left_hand_side_descriptors: the left hand side of equations as a list of :class:`Descriptor` objects
                                           (e.g. list of :class:`Moment`)
        :param right_hand_side: the right hand side of equations
        :param constants: the constants of the model
        """

        self.__left_hand_side_descriptors = left_hand_side_descriptors
        self.__left_hand_side = to_sympy_column_matrix(to_sympy_matrix([plhs.symbol for plhs in left_hand_side_descriptors]))
        self.__right_hand_side = to_sympy_column_matrix(right_hand_side)
        self.__constants = to_list_of_symbols(constants)
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

    # TODO: I don't think species_* methods should be part of ODEProblem, better for it to be unaware of description meanings
    @property
    def species_terms(self):
        return filter(lambda x: isinstance(x[1], Moment) and x[1].order == 1, self._descriptions_dict.iteritems())

    @property
    def number_of_species(self):
        return len(self.species_terms)

    @property
    def right_hand_side(self):
        return self.__right_hand_side

    @property
    def constants(self):
        return self.__constants

    @property
    def number_of_parameters(self):
        return len(self.constants)

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
        all_symbols = self.constants + self.variables
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
        Given the symbol associated with the problem, return the descriptor associated with that symbol

        :param symbol: Symbol
        :type symbol: basestring|:class:`sympy.Symbol`
        :return:
        """
        if isinstance(symbol, basestring):
            symbol = sympy.Symbol(symbol)

        try:
            return self._descriptions_dict[symbol]
        except KeyError:
            raise KeyError("Symbol {0!r} not found in left-hand-side of the equations".format(symbol))



    def __unicode__(self):
        equations_pretty_str = '\n\n'.join(['{0!r}:\n    {1!r}'.format(x, y) for x, y in zip(self.left_hand_side_descriptors,
                                                                                           self.right_hand_side)])
        return u"{0.__class__!r}\n" \
               u"Method: {0.method!r}\n" \
               u"Constants: {0.constants!r}\n" \
               u"\n" \
               u"Equations:\n\n" \
               u"{1}\n".format(self, equations_pretty_str)

    def __str__(self):
        return unicode(self).encode("utf8")

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
        lines.append("<p>Constants: <code>{0!r}</code></p>".format(self.constants))
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
        preamble = ["\documentclass{article}"]
        preamble += ["\usepackage[landscape, margin=0.5in, a3paper]{geometry}"]
        lines = ["\\begin{document}"]
        lines += ["\section*{%s}" % STRING_RIGHT_HAND]

        lines += ["$\dot {0} = {1} {2}$".format(str(sympy.latex(lhs.symbol)), str(sympy.latex(rhs)), r"\\")
                    for (rhs, lhs) in zip(self.right_hand_side, left_hand_side)]

        lines += [r"\\"] * 5

        lines += ["\section*{%s}" % STRING_MOM]


        lines += ["$\dot {0}$: {1} {2}".format(str(sympy.latex(lhs.symbol)), str(lhs), r"\\")
                       for lhs in left_hand_side if isinstance(lhs, Moment)]

        lines += ["\end{document}"]

        return '\n'.join(preamble + lines)


    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.constants == other.constants \
                   and other.left_hand_side_descriptors == self.left_hand_side_descriptors \
                   and sympy_expressions_equal(other.right_hand_side, self.right_hand_side)

    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = [('method', data.method),
                   ('constants', map(str, data.constants)),
                   ('left_hand_side_descriptors', list(data.left_hand_side_descriptors)),
                   ('right_hand_side', map(str, data.right_hand_side))]

        return dumper.represent_mapping(cls.yaml_tag, mapping)