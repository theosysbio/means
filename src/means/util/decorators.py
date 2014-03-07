from functools import wraps

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
