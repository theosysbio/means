from luigi.parameter import *

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

class DictParameter(Parameter):
    """
    A parameter that can take a dictionary of values. Most useful as a parameter to pass to **kwargs result.

    """

    _separator = ','
    _separator_key_value = ':'

    def parse(self, x):
        items = x.split(self._separator)

        key_values = []
        for item in items:
            if self._separator_key_value not in item:
                raise ValueError('Cannot split {0!r} into key,value pairs'.format(item))
            key_values.append(item.split(self._separator_key_value))

        return dict(key_values)

    def serialize(self, x):

        # Always sort the dictionary items before serialising, otherwise order is unspecified
        items = sorted(x.items())

        str_items = []
        for item in items:
            str_items.append(self._separator_key_value.join(item))

        return self._separator.join(str_items)


