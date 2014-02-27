import yaml
import numpy as np

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

_NUMPY_ARRAY_TAG = '!nparray'

#-- Special dump functions --------------------------------------------------------------------------------------------
class SerialisableObject(yaml.YAMLObject):

    @classmethod
    def from_yaml(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        return cls(**mapping)

    def to_file(self, file_):
        """
        Save the object to the file, specified by filename `file_` or the buffer provided in `file_`
        :param file_: filename of the file to save the object to, or already open file buffer
        """
        if isinstance(file_, basestring):
            file_ = open(file_, 'w')
            we_opened = True
        else:
            we_opened = False

        try:
            file_.write(dump(self))
        finally:
            if we_opened:
                file_.close()

    @classmethod
    def from_file(cls, file_):
        """
        Create new instance of the object from the file.
        :param file_: the filename of the file to read from or already opened file buffer
        """
        if isinstance(file_, basestring):
            file_ = open(file_, 'r')
            we_opened = True
        else:
            we_opened = False

        try:
            object_ = load(file_.read())
        finally:
            if we_opened:
                file_.close()

        if not isinstance(object_, cls):
            raise ValueError('Expected to read {0!r} object, but got {1!r} instead'.format(cls, type(object_)))

        return object_


class MeansDumper(Dumper):
    def __init__(self, stream, default_style=None, default_flow_style=None, canonical=None, indent=None, width=None,
                 allow_unicode=None, line_break=None, encoding=None, explicit_start=None, explicit_end=None,
                 version=None, tags=None):
        super(MeansDumper, self).__init__(stream, default_style, default_flow_style, canonical, indent, width,
                                          allow_unicode, line_break, encoding, explicit_start, explicit_end, version,
                                          tags)

        self.add_representer(np.ndarray, _ndarray_representer)
        self.add_multi_representer(np.float, _numpy_float_representer)


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

def _numpy_float_representer(dumper, data):
    """
    Get rid of the annoying representations of simple numbers, ie::

        !!python/object/apply:numpy.core.multiarray.scalar
          - !!python/object/apply:numpy.dtype
            args: [f8, 0, 1]
            state: !!python/tuple [3, <, null, null, null, -1, -1, 0]
          - !!binary |
            /Knx0k1iQD8=

    :param dumper:
    :param data:
    :return:
    """
    return dumper.represent_data(float(data))
