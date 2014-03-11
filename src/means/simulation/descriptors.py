"""
Descriptors that are local to the simulation package
"""
from means.core import Descriptor


class SensitivityTerm(Descriptor):
    r"""
    A :class:`~means.approximation.ode_problem.Descriptor` term that describes a particular object represents the sensitivity
    of some ODE term with respect to some parameter.
    In other words, sensitivity term describes :math:`s_{ij}(t) = \frac{\partial y_i(t)}{\partial p_j}` where
    :math:`y_i` is the ODE term described above and :math:`p_j` is the parameter.

    This class is used to describe sensitivity trajectories returned by :class:`means.simulation.simulate.Simulation`
    """
    _ode_term = None
    _parameter = None

    yaml_tag = '!sensitivity-term'

    def __init__(self, ode_term, parameter):
        """

        :param ode_term: the ode term whose sensitivity is being computed
        :type ode_term: :class:`~means.approximation.ode_problem.ODETermBase`
        :param parameter: parameter w.r.t. which the sensitivity is computed
        :type parameter: :class:`sympy.Symbol`
        """
        self._ode_term = ode_term
        self._parameter = parameter

    @property
    def ode_term(self):
        return self._ode_term

    @property
    def parameter(self):
        return self._parameter

    def __repr__(self):
        return '<Sensitivity of {0!r} w.r.t. {1!r}>'.format(self.ode_term, self.parameter)

    def __mathtext__(self):
        # Double {{ and }} in multiple places as to escape the curly braces in \frac{} from .format
        return r'$\frac{{\partial {0}}}{{\partial {1}}}$'.format(self.ode_term.symbol, self.parameter)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.ode_term == other.ode_term and self.parameter == other.parameter

    @classmethod
    def to_yaml(cls, dumper, data):
        mapping = [('ode_term', data.ode_term),
                   ('parameter', data.parameter)]

        return dumper.represent_mapping(cls.yaml_tag, mapping)


class PerturbedTerm(Descriptor):
    r"""
    A :class:`~means.approximation.ode_problem.Descriptor` term that describes a particular object represents the sensitivity
    of some ODE term with respect to some parameter.
    In other words, sensitivity term describes :math:`s_{ij}(t) = \frac{\partial y_i(t)}{\partial p_j}` where
    :math:`y_i` is the ODE term described above and :math:`p_j` is the parameter.

    This class is used to describe sensitivity trajectories returned by :class:`means.simulation.simulate.Simulation`
    """
    _ode_term = None
    _parameter = None
    _delta = None

    def __init__(self, ode_term, parameter, delta=0.01):
        """

        :param ode_term: the ode term whose sensitivity is being computed
        :type ode_term: :class:`~means.approximation.ode_problem.ODETermBase`
        :param parameter: parameter w.r.t. which the sensitivity is computed
        :type parameter: :class:`sympy.Symbol`
        """
        self._ode_term = ode_term
        self._parameter = parameter
        self._delta = delta

    @property
    def ode_term(self):
        return self._ode_term

    @property
    def parameter(self):
        return self._parameter

    @property
    def delta(self):
        return self._delta

    def __repr__(self):
        return '<Perturbed {0!r} when {1!r} is perturbed by {2!r}>'.format(self.ode_term, self.parameter, self.delta)

    def __mathtext__(self):
        # Double {{ and }} in multiple places as to escape the curly braces in \frac{} from .format
        return r'${0}$ when ${1}={1}+{2}$'.format(self.ode_term.symbol, self.parameter, self.delta)