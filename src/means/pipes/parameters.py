from luigi.parameter import *

class ListParameter(Parameter):

    _separator = ','

    def __init__(self, item_type=None, *args, **kwargs):
        self._item_type = item_type
        super(ListParameter, self).__init__(*args, **kwargs)

    def parse(self, x):
        if isinstance(x, basestring):
            items = x.split(',')
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