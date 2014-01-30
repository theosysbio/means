import sympy
from sympy.core.sympify import SympifyError

def substitute_all(expr, pairs):
    """
    Performs multiple substitutions in an expression
    :param expr: a sympy expression
    :param pairs: a list of pairs (a,b) where each b_i is to be substituted with a_i
    :return: the substituted expression
    """
    out = expr
    for (a,b) in pairs:
        out = sympy.Subs(out, b, a)
    to_ret = out.doit()
    return to_ret

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


def sympy_expressions_equal(expr1, expr2):
    """
    Compare two sympy expressions that are not necessarily expanded.
    :param expr1: a first expression
    :param expr2: a second expression
    :return: True if the expressions are similar, False otherwise
    """
    # the simplified difference is equal to zero: same expressions

    try:
        return sympy.simplify(expr1 - expr2) == 0
    except SympifyError:
        # Doing sympy.simplify(expr1 - expr2) raises an error if expr1 or expr2 is a matrix
        if isinstance(expr1, sympy.Matrix) or isinstance(expr2, sympy.Matrix):
            return _sympy_matrices_equal(expr1, expr2)
        else:
            raise

def assert_sympy_expressions_equal(expr1, expr2):
    """
    Raises `AssertionError` if `expr1` is not equal to `expr2`.

    :param expr1: first expression
    :param expr2: second expression
    :return: None
    """
    if not sympy_expressions_equal(expr1, expr2):
        raise AssertionError("{0!r} != {1!r}".format(expr1, expr2))

def _sympy_matrices_equal(matrix_left, matrix_right):
    """
    Compare two sympy matrices that are not necessarily expanded.
    Calls `deep_compare_expressions` for each element in the matrices.

    Private function. Use `sympy_expressions_equal`.
    The former should be able to compare everything.

    :param matrix_left:
    :param matrix_right:
    :return:
    """
    if matrix_left.cols != matrix_right.cols or matrix_left.rows != matrix_right.rows:
        return False

    for expression_left, expression_right in zip(matrix_left, matrix_right):
        if not sympy_expressions_equal(expression_left, expression_right):
            return False

    return True

