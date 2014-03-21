"""
MEA helper functions.
-------

This part of the package provides a few small utility
functions for the rest :mod:`~means.approximation.mea`.
"""

import operator
import sympy as sp
from means.util.decorators import cache
from means.util.sympyhelpers import product


@cache
def _cached_diff(expression, var):
    """
    Derive expression with respect to a single variable.

    :param expression: an expression to derive
    :type expression: :class:`~sympy.Expr`
    :param var: a variable
    :type var: :class:`~sympy.Symbol`
    :return: the derived expression
    :type: :class:`~sympy.Expr`
    """
    return sp.Derivative(expression, var, evaluate=True)


@cache
def get_one_over_n_factorial(counter_entry):
    r"""
    Calculates  the :math:`\frac{1}{\mathbf{n!}}` of eq. 6 (see Ale et al. 2013).
    That is the invert of a product of factorials.
    :param counter_entry: an entry of counter. That is an array of integers of length equal to the number of variables.
    For instance, `counter_entry` could be `[1,0,1]` for three variables.
    :return: a scalar as a sympy expression
    """
    # compute all factorials
    factos = [sp.factorial(c) for c in counter_entry]
    # multiply them
    prod = product(factos)
    # return the invert
    return sp.Integer(1)/sp.S(prod)


def derive_expr_from_counter_entry(expression, species, counter_entry):
    r"""
    Derives an given expression with respect to arbitrary species and orders.
    This is used to compute :math:`\frac{\partial^n \mathbf{n}a_l(\mathbf{x})}{\partial \mathbf{x^n}}` in eq. 6

    :param expression: the expression to be derived
    :type expression: :class:`~sympy.Expr`
    :param species: the name of the variables (typically {y_0, y_1, ..., y_n})
    :type species: list[:class:`~sympy.Symbol`]
    :param counter_entry: an entry of counter. That is a tuple of integers of length equal to the number of variables.
    For example, (0,2,1) means we derive with respect to the third variable (first order)
    and to the second variable (second order)

    :return: the derived expression
    """

    # no derivation, we return the unchanged expression
    if sum(counter_entry) == 0:
        return expression

    # repeat a variable as many time as its value in counter
    diff_vars = reduce(operator.add, map(lambda v, c: [v] * c, species, counter_entry))
    out_expr = expression

    for var in diff_vars:
        # If the derivative is already 0, we can return 0
        if out_expr.is_Integer:
            return sp.Integer(0)
        out_expr = _cached_diff(out_expr, var)

    return out_expr


def make_k_chose_e(e_vec, k_vec):
    """
    Computes the product :math:`{\mathbf{n} \choose \mathbf{k}}`

    :param e_vec: the vector e
    :type e_vec: :class:`numpy.array`
    :param k_vec: the vector k
    :type k_vec: :class:`numpy.array`
    :return: a scalar
    """
    return product([sp.factorial(k) / (sp.factorial(e) * sp.factorial(k - e)) for e,k in zip(e_vec, k_vec)])