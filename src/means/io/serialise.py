import yaml
import numpy as np

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
class SerialisableObject(yaml.YAMLObject):

    @classmethod
    def from_yaml(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        return cls(**mapping)


class MeansDumper(Dumper):
    def __init__(self, stream, default_style=None, default_flow_style=None, canonical=None, indent=None, width=None,
                 allow_unicode=None, line_break=None, encoding=None, explicit_start=None, explicit_end=None,
                 version=None, tags=None):
        super(MeansDumper, self).__init__(stream, default_style, default_flow_style, canonical, indent, width,
                                          allow_unicode, line_break, encoding, explicit_start, explicit_end, version,
                                          tags)

        self.add_representer(np.ndarray, _ndarray_representer)


class MeansLoader(Loader):

    def __init__(self, stream):
        super(MeansLoader, self).__init__(stream)
        self.add_constructor(_NUMPY_ARRAY_TAG, _generic_constructor(np.array))


def dump(object_):
    return yaml.dump(object_, Dumper=MeansDumper)


def load(data):
    return yaml.load(data, Loader=MeansLoader)

#-- Generic constructor ------------------------------

def _generic_constructor(class_):
    def f(loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        return class_(**mapping)
    return f

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
