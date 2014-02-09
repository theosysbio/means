import sympy as sp
import itertools
import operator
from means.util.sympyhelpers import product

def derive_expr_from_counter_entry(expression, species, counter_entry):
    """
    Derives an given expression with respect to arbitrary species and orders.
    This is used to compute :math:`\frac{\partial^n \mathbf{n}a_l(\mathbf{x})}{\partial \mathbf{x^n}}` in eq. 6
    :param expression: the expression to be derived
    :param species: the name of the variables (typically {y_0, y_1, ..., y_n})
    :param counter_entry: an entry of counter. That is a tuple of integers of length equal to the number of variables.
    For example, (0,2,1) means we derive with respect to the third variable (first order)
    and to the second variable (second order)

    :return: the derived expression
    """
    # no derivation, we return the unchanged expression
    if sum(counter_entry) == 0:
        return expression
    # repeat a variable as many time as its value in counter
    diff_orders = reduce(operator.add, map(lambda v, c: [v] * c, species, counter_entry))
    # pass this as arguments for sympy diff
    return sp.diff(expression, *diff_orders)

def get_factorial_term(counter_entry):
    """
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

def generate_dmu_over_dt(species, propensity, n_counter, stoichiometry_matrix):
    """
    Calculate :math:`\frac{d\mu_i}{dt}` in eq. 6 (see Ale et al. 2013).
    .. math::
         \frac{d\mu_i}{dt} = S \begin{bmatrix} \sum_{l} \sum_{n_1=0}^{\infty} ... \sum_{n_d=0}^{\infty} \frac{1}{\mathbf{n!}}\frac{\partial^n \mathbf{n}a_l(\mathbf{x})}{\partial \mathbf{x^n}} \|_{x=\mu} \mathbf{M_{x^n}} \end{bmatrix}

    :param species: the name of the variables (typically {y_0, y_1, ..., y_n})
    :param propensity: the reactions describes by the model
    :param n_counter: a list of central moments
    :param stoichiometry_matrix: the stoichiometry matrix
    :return: a matrix in which each row corresponds to a reaction, and each column to an element of counter.
    """

    # compute derivatives :math:`\frac{\partial^n \mathbf{n}a_l(\mathbf{x})}{\partial \mathbf{x^n}}`
    # for EACH REACTION and EACH entry in COUNTER
    derivs =[derive_expr_from_counter_entry(reac, species, c.n_vector) for (reac, c) in itertools.product(propensity, n_counter)]
    # Computes the factorial terms (:math:`\frac{1}{\mathbf{n!}}`) for EACH REACTION and EACH entry in COUNTER
    # this does not depend of the reaction, so we just repeat the result for each reaction
    factorial_terms = [get_factorial_term(c.n_vector) for (c) in n_counter] * len(propensity)
    # we make a matrix in which every element is the entry-wise multiplication of `derives` and factorial_terms
    taylor_exp_matrix = sp.Matrix(len(propensity), len(n_counter), [d*f for (d, f) in zip(derivs, factorial_terms)])
    # dmu_over_dt is the product of the stoichiometry matrix by the taylor expansion matrix
    return stoichiometry_matrix * taylor_exp_matrix
