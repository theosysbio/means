import sympy
from sympy.core.sympify import SympifyError
import numpy as np


def substitute_all(sp_object, pairs):
    """
    Performs multiple substitutions in an expression
    :param expr: a sympy matrix or expression
    :param pairs: a list of pairs (a,b) where each a_i is to be substituted with b_i
    :return: the substituted expression
    """
    # we recurse if the object was a matrix so we apply substitution to all elements
    if isinstance(sp_object, sympy.Matrix):
        return sp_object.applyfunc(lambda x: substitute_all(x, pairs))

    try:
        expr = sp_object.subs(pairs)

    # in sympy 0.7.2, this would not work, so we do it manually
    except:
        expr =  sp_object
        for (a,b) in pairs:
            expr = sympy.Subs(expr, a, b)
        expr = expr.doit()
    return expr


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
    try:
        return sympy.Matrix(value)
    except ValueError as original_exception:
        if isinstance(value, list) and len(value) and all([not isinstance(x, list) for x in value]):
            # Let's try to convert the contents into a list
            # (this is required for 0.7.4 if we pass in a list of strings)
            # See `test_creation_of_column_matrix_from_list_of_strings` test
            list_of_lists_value = [[x] for x in value]
            try:
                m = sympy.Matrix(list_of_lists_value)
                return m
            except Exception:
                raise original_exception
        else:
            raise original_exception




def to_sympy_column_matrix(matrix):
    """
    Converts a sympy matrix to a column matrix (i.e. transposes it if it was row matrix)
    Raises ValueError if matrix provided is not a vector
    :param matrix: a vector to be converted to column
    :return:
    """
    matrix = to_sympy_matrix(matrix)
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
        difference = sympy.simplify(expr1 - expr2)
    except SympifyError:
        # Doing sympy.simplify(expr1 - expr2) raises an error if expr1 or expr2 is a matrix (for sympy 0.7.2)
        if isinstance(expr1, sympy.Matrix) or isinstance(expr2, sympy.Matrix):
            return _sympy_matrices_equal(expr1, expr2)
        else:
            raise

    # sympy 0.7.4 returns matrix of zeros for equal matrices
    try:
        difference = sum(difference)
    except TypeError:
        pass

    return difference == 0


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



def _eval_res_equal(expr1, expr2, atoms, vals,threshold=10e-10):

    """
    Compare two expressions after evaluation of symbols by random expressions

    private function called by `sympy_empirical_equal`

    :param expr1: a first sympy expression
    :param expr2: a second sympy expression
    :param atoms: the component symbols (they are assumed to be common)
    :param vals: the values to subtitute atoms with
    :return: True if the result is the same
    """
    substitution_pairs = zip(atoms, vals)

    eval_1 = substitute_all(expr1, substitution_pairs)
    eval_2 = substitute_all(expr2, substitution_pairs)

    return abs(eval_1 - eval_2) < threshold


def sympy_empirical_equal(expr1, expr2):

    """
    Compare long , complex, expressions by replacing all symbols by a set of arbitrary expressions

    :param expr1: first expression
    :param expr2: second expression
    :return: True if expressions are empirically equal, false otherwise
    """

    atoms_1 = expr1.atoms()
    atoms_1 = [a for a in atoms_1 if isinstance(a,sympy.Symbol)]
    atoms_2 = expr2.atoms()
    atoms_2 = [a for a in atoms_2 if isinstance(a,sympy.Symbol)]


    # lets us merge symbol in case one equation has more / different symbols
    atoms = set(atoms_1 + atoms_2)


    arbitrary_values = []
    arbitrary_values.append([i * 7.1 for i in range(1,len(atoms)+1)])
    arbitrary_values.append([i / 9.3 for i in range(1,len(atoms)+1)])
    arbitrary_values.append([i ** 2 for i in range(1,len(atoms)+1)])

    # test for arbitrary value. return false at the first failure
    for av in arbitrary_values:
        if not _eval_res_equal(expr1, expr2, atoms, av):
            return False

    return True

def sum_of_rows(mat):
    out = sympy.Matrix([sum(r)for r in mat.tolist()])
    return out

def sum_of_cols(mat):
    out = sympy.Matrix([sum(r)for r in (mat.T).tolist()]).T
    return out

import operator

def product(list):
    return reduce(operator.mul, list)

def sympy_sum_list(list):
    return reduce(operator.add, list)

def to_one_dim_array(iterable_, dtype=np.double):
    return np.array(iterable_, dtype = dtype).reshape(-1,)
