import sympy
def to_sympy_matrix(value):
    """
    Converts value to a `sympy.Matrix` object, if possible.
    Leaves the value as `sympy.Matrix` if it already was
    :param value: value to convert
    :return:
    :rtype: sympy.Matrix
    """
    if isinstance(value, sympy.Matrix):
        return value
    return sympy.Matrix(value)

def to_sympy_column_matrix(matrix):
    """
    Converts a sympy matrix to a column matrix (i.e. transposes it if it was row matrix)
    Raises ValueError if matrix provided is not a vector
    :param matrix: a vector to be converted to column
    :return:
    """
    if matrix.cols == 1:
        return matrix
    elif matrix.rows == 1:
        return matrix.T
    else:
        raise ValueError('Cannot convert {0!r} to a column matrix'.format(matrix))

def to_list_of_symbols(values):
    if isinstance(values, sympy.Matrix):
        # If we got a matrix, convert it to a list
        values = list(values)
    # Convert list to a list of symbols
    return sympy.sympify(values)