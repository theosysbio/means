import means
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


_MODEL_TAG = '!model'

class MeansDumper(Dumper):
    def __init__(self, stream, default_style=None, default_flow_style=None, canonical=None, indent=None, width=None,
                 allow_unicode=None, line_break=None, encoding=None, explicit_start=None, explicit_end=None,
                 version=None, tags=None):
        super(MeansDumper, self).__init__(stream, default_style, default_flow_style, canonical, indent, width,
                                          allow_unicode, line_break, encoding, explicit_start, explicit_end, version,
                                          tags)

        self.add_representer(means.Model, _model_representer)

class MeansLoader(Loader):

    def __init__(self, stream):
        super(MeansLoader, self).__init__(stream)
        self.add_constructor(_MODEL_TAG, _model_constructor)


def dump(object):
    return yaml.dump(object, Dumper=MeansDumper)

def load(data):
    return yaml.load(data, Loader=MeansLoader)

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

def _model_constructor(loader, node):
    """
    A helper to be able to load the model object hat have been dumped

    :param loader: `loader` instance passed in from :mod:`yaml`
    :param node: already parsed :mod:`yaml` node
    :return: parsed Model object
    :rtype: :class:`means.Model`
    """
    mapping = loader.construct_mapping(node, deep=True)
    return means.Model(**mapping)