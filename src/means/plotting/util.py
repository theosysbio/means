def mathtextify(item):
    """
    Tries to convert the item to text, via the `__mathtext__` method defined in the item.
    If such method does not exist, does nothing and just returns string version of item

    :param item: item to convert to mathtext specification
    :return: mathtextified string
    """

    try:
        return item.__mathtext__()
    except AttributeError:
        return unicode(item)