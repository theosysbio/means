#todo docsting. wait for decision on geting rid of variance term -> moment corresponding to variance


import numpy as np
import sympy
from means.io.serialise import SerialisableObject


class Descriptor(SerialisableObject):
    yaml_tag = u"!descriptor"

    def mathtext(self):
        return str(self)

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

    def mathtext(self):
        # Double {{ and }} in multiple places as to escape the curly braces in \frac{} from .format
        return r'${0}$'.format(sympy.printing.latex(self.symbol))

    def _repr_latex(self):
        return '${0}$'.format(self.symbol)


class VarianceTerms(ODETermBase):
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
        super(VarianceTerms, self).__init__(symbol=symbol)
        self._position = position

    @property
    def position(self):
        return self._position

    def __unicode__(self):
        return u'{0}(position{1}, symbol={2})'.format(self.__class__.__name__, self.position, self.symbol)

    def _repr_latex_(self):
        return '(Variance term $V_{{{0}, {1}}} ${2}$)'.format(self.position[0], self.position[1], self.symbol)

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

    def __hash__(self):
        # Allows moment objects to be stored as keys to dictionaries
        return hash(repr(self.n_vector))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return np.equal(self.n_vector, other.n_vector).all() and self.symbol == other.symbol

    def __ne__(self, other):
        return not other == self

    def __ge__(self, other):
        """
        A moment is said greater or equal than another iff all the values of n_vec are greater or equal.
        Mathematically: ::math::`n_i^a \ge n_i^b ~ \textrm{for all i}`
        """
        return (self.n_vector >= other.n_vector).all()

    def __unicode__(self):
        return u'{self.__class__.__name__}({self.n_vector!r}, symbol={self.symbol!r})'.format(self=self)

    def __str__(self):
        return unicode(self).encode("utf8")

    def __repr__(self):
        return str(self)

    def _repr_latex_(self):
        return '{0}($[{1}]$, symbol=${2}$)'.format(self.__class__.__name__, ', '.join(map(str, self.n_vector)), self.symbol)

    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = [('symbol', str(data.symbol)), ('n_vector', data.n_vector.tolist())]
        return dumper.represent_mapping(cls.yaml_tag, mapping)