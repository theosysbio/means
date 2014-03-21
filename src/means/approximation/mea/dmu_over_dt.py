import itertools
import sympy as sp
from means.approximation.mea.mea_helpers import get_one_over_n_factorial, derive_expr_from_counter_entry


def generate_dmu_over_dt(species, propensity, n_counter, stoichiometry_matrix):
    r"""
    Calculate :math:`\frac{d\mu_i}{dt}` in eq. 6 (see Ale et al. 2013).

    .. math::
         \frac{d\mu_i}{dt} = S \begin{bmatrix} \sum_{l} \sum_{n_1=0}^{\infty} ...
         \sum_{n_d=0}^{\infty}
         \frac{1}{\mathbf{n!}}
         \frac{\partial^n \mathbf{n}a_l(\mathbf{x})}{\partial \mathbf{x^n}} |_{x=\mu}
         \mathbf{M_{x^n}} \end{bmatrix}

    :param species: the name of the species/variables (typically `['y_0', 'y_1', ..., 'y_n']`)
    :type species: list[`sympy.Symbol`]
    :param propensity: the reactions describes by the model
    :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
    :type n_counter: list[:class:`~means.core.descriptors.Moment`]
    :param stoichiometry_matrix: the stoichiometry matrix
    :type stoichiometry_matrix: `sympy.Matrix`
    :return: a matrix in which each row corresponds to a reaction, and each column to an element of counter.
    """


    # compute derivatives :math:`\frac{\partial^n \mathbf{n}a_l(\mathbf{x})}{\partial \mathbf{x^n}}`
    # for EACH REACTION and EACH entry in COUNTER
    derives =[derive_expr_from_counter_entry(reac, species, c.n_vector) for (reac, c) in itertools.product(propensity, n_counter)]
    # Computes the factorial terms (:math:`\frac{1}{\mathbf{n!}}`) for EACH REACTION and EACH entry in COUNTER
    # this does not depend of the reaction, so we just repeat the result for each reaction
    factorial_terms = [get_one_over_n_factorial(tuple(c.n_vector)) for c in n_counter] * len(propensity)
    # we make a matrix in which every element is the entry-wise multiplication of `derives` and factorial_terms
    taylor_exp_matrix = sp.Matrix(len(propensity), len(n_counter), [d*f for (d, f) in zip(derives, factorial_terms)])
    # dmu_over_dt is the product of the stoichiometry matrix by the taylor expansion matrix
    return stoichiometry_matrix * taylor_exp_matrix
