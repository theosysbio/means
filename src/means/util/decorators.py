from functools import wraps

def memoised_property(function):

    @wraps(function)
    def wrapper(self):
        try:
            return self._memoised_properties[function]
        except AttributeError:
            self._memoised_properties = {function: function(self)}
            return self._memoised_properties[function]
        except KeyError:
            self._memoised_properties[function] = function(self)
            return self._memoised_properties[function]

    return property(wrapper)


def cache(func):
    cache = {}
    @wraps(func)
    def wrap(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrap

# def cache(obj):
#     cache = obj.cache = {}
#
#     @wraps(obj)
#     def memoizer(*args, **kwargs):
#         key = str(args) + str(kwargs)
#         if key not in cache:
#             cache[key] = obj(*args, **kwargs)
#         return cache[key]
#     return memoizer
