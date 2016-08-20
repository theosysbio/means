import sys
import itertools

PY2 = sys.version_info[0] < 3

try:
    # py2
    from StringIO import StringIO
except ImportError:
    # py3
    from io import StringIO

try:
    # py2
    reduce = reduce
except NameError:
    # py3
    from functools import reduce

try:
    # py2
    string_types = basestring
    text_type = unicode
except NameError:
    # py3
    string_types = (str,)
    text_type = str

try:
    # py2
    zip = itertools.izip
except AttributeError:
    # py3
    pass # builtins.zip already equivalent to py2 itertools.izip

def unicode_compatible(cls):
    if PY2:
        cls.__unicode__ = cls.__str__
        cls.__str__ = lambda self: self.__unicode__().encode('utf8')
    return cls

try:
    dict.iteritems
except AttributeError:
    # py3
    def iteritems(d):
        for item in d.items():
            yield item
else:
    # py2
    def iteritems(d):
        for item in d.iteritems():
            yield item
