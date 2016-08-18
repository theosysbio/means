from __future__ import absolute_import, print_function

from functools import wraps

def memoised_property(function):

    @wraps(function)
    def wrapper(self):
        if not isinstance(self, MemoisableObject):
            raise TypeError('Can only memoise properties of MemoisableObject instances')

        try:
            return self._memoised_properties[function]
        except AttributeError:
            self._memoised_properties = {function: function(self)}
            return self._memoised_properties[function]
        except KeyError:
            self._memoised_properties[function] = function(self)
            return self._memoised_properties[function]

    return property(wrapper)


class MemoisableObject(object):
    """
    A wrapper around objects that support ``memoised_property`` decorator.
    It overrides the __getstate__ method to prevent pickling cached values.
    """

    def __getstate__(self):
        state = self.__dict__.copy()

        # Remove the cached stuff as this will prevent pickling
        try:
            del state['_memoised_properties']
        except KeyError:
            pass

        return state
