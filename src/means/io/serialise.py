import sympy
import means
import yaml
import numpy as np
import means.approximation.ode_problem

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


_MODEL_TAG = '!model'
_ODE_PROBLEM_TAG = '!problem'
_NUMPY_ARRAY_TAG = '!nparray'
_MOMENT_TAG = '!moment'
_VARIANCE_TERM_TAG = '!variance-term'
_TRAJECTORY_TAG = '!trajectory'
_TRAJECTORY_WITH_SENSITIVITY_TAG = '!trajectory-with-sensitivities'
_SENSITIVITY_TERM_TAG = '!sensitivity-term'

#-- Special dump functions --------------------------------------------------------------------------------------------
class MeansDumper(Dumper):
    def __init__(self, stream, default_style=None, default_flow_style=None, canonical=None, indent=None, width=None,
                 allow_unicode=None, line_break=None, encoding=None, explicit_start=None, explicit_end=None,
                 version=None, tags=None):
        super(MeansDumper, self).__init__(stream, default_style, default_flow_style, canonical, indent, width,
                                          allow_unicode, line_break, encoding, explicit_start, explicit_end, version,
                                          tags)

        self.add_representer(np.ndarray, _ndarray_representer)
        self.add_representer(means.Model, _model_representer)
        self.add_representer(means.approximation.ODEProblem, _problem_representer)
        self.add_representer(means.approximation.ode_problem.Moment, _moment_representer)
        self.add_representer(means.approximation.ode_problem.VarianceTerm, _variance_term_representer)
        self.add_representer(means.simulation.SensitivityTerm, _sensitivity_term_representer)
        self.add_representer(means.simulation.Trajectory, _trajectory_representer)
        self.add_representer(means.simulation.TrajectoryWithSensitivityData,
                             _trajectory_with_sensitivity_data_representer)


class MeansLoader(Loader):

    def __init__(self, stream):
        super(MeansLoader, self).__init__(stream)
        self.add_constructor(_MODEL_TAG, _generic_constructor(means.Model))
        self.add_constructor(_NUMPY_ARRAY_TAG, _generic_constructor(np.array))
        self.add_constructor(_ODE_PROBLEM_TAG, _generic_constructor(means.approximation.ODEProblem))
        self.add_constructor(_MOMENT_TAG, _generic_constructor(means.approximation.ode_problem.Moment))
        self.add_constructor(_VARIANCE_TERM_TAG, _generic_constructor(means.approximation.ode_problem.VarianceTerm))
        self.add_constructor(_TRAJECTORY_TAG, _generic_constructor(means.simulation.Trajectory))
        self.add_constructor(_TRAJECTORY_WITH_SENSITIVITY_TAG,
                             _generic_constructor(means.simulation.TrajectoryWithSensitivityData))
        self.add_constructor(_SENSITIVITY_TERM_TAG, _generic_constructor(means.simulation.SensitivityTerm))

def dump(object):
    return yaml.dump(object, Dumper=MeansDumper)

def load(data):
    return yaml.load(data, Loader=MeansLoader)

def _generic_constructor(class_):
    def f(loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        return class_(**mapping)
    return f

#-- Representers/Constructors for Model objects ------------------------------------------------------------------------

def _model_representer(dumper, data):
    """
    A helper to nicely dump model objects

    :param dumper: `dumper` instance that would be passed by `yaml`
    :param data: data to serialise, a :class:`means.Model` object
    :type data: :class:`means.Model`
    :return: Serialised string
    :rtype: str
    """
    mapping = [('species', map(str, data.species)), ('constants', map(str, data.constants)),
               ('stoichiometry_matrix', map(lambda x: map(int, x), data.stoichiometry_matrix.tolist())),
               ('propensities', map(str, data.propensities))]

    return dumper.represent_mapping(_MODEL_TAG, mapping)

#-- Representers/Constructors for numpy objects ------------------------------------------------------------------------
def _ndarray_representer(dumper, data):
    """

    :param dumper:
    :param data:
    :type data: :class:`numpy.ndarray`
    :return:
    """
    mapping = [('object', data.tolist()), ('dtype', data.dtype.name)]
    return dumper.represent_mapping(_NUMPY_ARRAY_TAG, mapping)

#-- Representers/Constructors for Descriptor objects ------------------------------------------------------------------
def _moment_representer(dumper, data):
    """
    Representer for moment object
    :param dumper:
    :param data:
    :type data: :class:`means.approximation.ode_problem.Moment`
    :return:
    """
    mapping = [('symbol', str(data.symbol)), ('n_vector', data.n_vector.tolist())]
    return dumper.represent_mapping(_MOMENT_TAG, mapping)

def _variance_term_representer(dumper, data):
    """
    Representer for moment object
    :param dumper:
    :param data:
    :type data: :class:`means.approximation.ode_problem.Moment`
    :return:
    """
    mapping = [('symbol', str(data.symbol)), ('position', data.position)]
    return dumper.represent_mapping(_VARIANCE_TERM_TAG, mapping)

def _sensitivity_term_representer(dumper, data):
    mapping = [('ode_term', data.ode_term),
               ('parameter', data.parameter)]

    return dumper.represent_mapping(_SENSITIVITY_TERM_TAG, mapping)

#-- Representers/Constructors for ODEProblem objects -------------------------------------------------------------------
def _problem_representer(dumper, data):
    """
    :param dumper:
    :param data:
    :type data: :class:`means.approximation.ODEProblem`
    :return:
    """

    mapping = [('method', data.method),
               ('constants', map(str, data.constants)),
               ('ode_lhs_terms', list(data.ode_lhs_terms)),
               ('right_hand_side', map(str, data.right_hand_side))]

    return dumper.represent_mapping(_ODE_PROBLEM_TAG, mapping)

#-- Represetners/Constructors for Trajectory objects ------------------------------------------------------------------
def _trajectory_representer(dumper, data):

    mapping = [('timepoints', data.timepoints),
               ('values', data.values),
               ('description', data.description)]
    return dumper.represent_mapping(_TRAJECTORY_TAG, mapping)

def _trajectory_with_sensitivity_data_representer(dumper, data):
    mapping = [('timepoints', data.timepoints),
               ('values', data.values),
               ('description', data.description),
               ('sensitivity_data', data.sensitivity_data)]
    return dumper.represent_mapping(_TRAJECTORY_WITH_SENSITIVITY_TAG, mapping)