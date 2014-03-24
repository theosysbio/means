from luigi.parameter import *
import means.examples

class ListParameter(Parameter):
    """
    A special parameter for lists, explicitly defined to be easily serialisable and deserialisable to
    human readable string
    """
    _separator = ','

    def __init__(self, item_type=None, *args, **kwargs):
        self._item_type = item_type
        super(ListParameter, self).__init__(*args, **kwargs)

    def parse(self, x):
        if isinstance(x, basestring):
            items = x.split(self._separator)
            if self._item_type is not None:
                items = map(self._item_type, items)
            return items
        else:
            if self._item_type is not None:
                return map(self._item_type, x)
            else:
                return list(x)

    def serialize(self, x):
        return ','.join(map(str, x))

class ListOfKeyValuePairsParameter(Parameter):
    """
    A parameter that can take a list of key-value pairs.
    Most useful to pass in kwargs for the function.

    Please use :meth:`dict.items()` before assigning this parameter.
    This is needed as dictionaries are not serialisable by default in python,
    therefore caching mechanisms will not work.

    Lists, on the other hand, are serialisable by :mod:`luigi` as they are always converted to tuples beforehand.
    """

    _separator = ','
    _separator_key_value = ':'

    def parse(self, x):
        if isinstance(x, basestring):
            items = x.split(self._separator)

            key_values = []
            for item in items:
                if self._separator_key_value not in item:
                    raise ValueError('Cannot split {0!r} into key,value pairs'.format(item))
                key_values.append(item.split(self._separator_key_value))

            return key_values
        elif isinstance(x, list):
            return x
        else:
            raise TypeError('Expected a list or string')

    def serialize(self, x):
        items = x

        str_items = []
        for item in items:
            str_items.append(self._separator_key_value.join(item))
        return self._separator.join(str_items)



class ModelParameter(Parameter):

    def parse(self, x):
        if isinstance(x, means.Model):
            return x
        else:
            raise TypeError('{0!r} is not a model name, nor a `means.Model` object')

    def serialize(self, x):
        return x
