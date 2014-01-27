from sympy import Matrix, Symbol, factorial
from sympy import S as F
import sympy as sp
import itertools
import operator

def derive_expr_from_counter_entry(expression, variables, counter_entry):

    """
    Derives an given expression with respect to arbitrary variables and orders.

    :param expression: the expression to be derived
    :param variables: the name of the variables (typically {y_0, y_1, ..., y_n})
    :param counter_entry: an entry of counter. That is a tuple of integers of length equal to the number of variables.
    For example, (0,2,1) means we derive with respect to the third variable (first order)
    and to the second variable (second order)

    :return: the derived expression
    """

    assert(len(variables) == len(counter_entry))

    # no derivation, we return the unchanged expression
    if sum(counter_entry) == 0:
        return expression


    expr_out = expression
    # we recursively derive the expression with respect to all variables at the degree specified in counter
    for (var, degree) in zip(variables,counter_entry):
        expr_out = sp.diff(expr_out, var, degree)
        # If the expression reaches 0, we can return 0
        if(expr_out == sp.S(0)):
            return sp.S(0)

    return expr_out

def get_factorial_term(counter_entry):

    """
    Calculates  the "1/n!" of eq. 6 (see Ale et al. 2013). Note that n! is a product of factorials.

    :param counter_entry: an entry of counter. That is a tuple of integers of length equal to the number of variables.
    For instance, (1,0,1) could exist for three variables.

    :return: the scalar result scalar as a sympy expression
    """

    # compute all factorials
    factos = [factorial(c) for c in counter_entry]
    # multiply them
    prod = reduce(operator.mul,factos)
    # return the invert
    return sp.S(1)/sp.S(prod)


def taylor_expansion (variables, propensity, counter):
    """
    Calculates  creates terms used in eq. 6 (see Ale et al. 2013) to calculate dmu/dt for EACH VARIABLE combination,
    and EACH REACTION.

    :param variables: the name of the variables (typically {y_0, y_1, ..., y_n})
    :param propensity: the reactions describes by the model
    :param counter: a list of all possible combination of order of derivation
    :return: a matrix in which each row corresponds to a reaction, and each column to an element of counter.
    """

    # compute derivatives for EACH REACTION and EACH entry in COUNTER
    derives =[derive_expr_from_counter_entry(reac, variables, c) for (reac, c) in itertools.product(propensity, counter)]

    # Computes the factorial terms for EACH REACTION and EACH entry in COUNTER
    # this does not depend of the reaction, so we just repeat the result for each reaction
    factorial_terms = [get_factorial_term(c) for (c) in counter] * len(propensity)

    # we make a matrix in which every element is the entry-wise multiplication of `derives` and factorial_terms
    te_matrix = sp.Matrix(len(propensity), len(counter), [d*f for (d,f) in zip(derives, factorial_terms)])
    return te_matrix


