from __future__ import absolute_import, print_function

import itertools

import sympy as sp

from means.approximation.mea.mea_helpers import get_one_over_n_factorial, derive_expr_from_counter_entry, make_k_chose_e
from means.util.sympyhelpers import sum_of_cols, product
from means.util.decorators import cache


class DBetaOverDtCalculator(object):
    """
    A class providing a efficient way to recursively calculate :math:`\\frac{d\\beta}{dt}` (eq. 11 in  [Ale2013]_).
    A class was used here merely for optimisation reasons.

    .. [Ale2013] A. Ale, P. Kirk, and M. P. H. Stumpf,\
       "A general moment expansion method for stochastic kinetic models,"\
       The Journal of Chemical Physics, vol. 138, no. 17, p. 174101, 2013.
    """
    def __init__(self, propensities, n_counter, stoichoimetry_matrix, species):
        """
        :param propensities:  the rates/propensities of the reactions
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param stoichoimetry_matrix: The stoichiometry matrix. Explicitly provided by the model
        :param species: the names of the variables/species

        """
        self.__propensities = propensities
        self.__n_counter = n_counter
        self.__stoichoimetry_matrix = stoichoimetry_matrix
        self.__species = tuple(species)

    def get(self, k_vec, e_counter):
        r"""
        Provides the terms needed for equation 11 (see Ale et al. 2013).
        This gives the expressions for :math:`\frac{d\beta}{dt}` in equation 9, these are the
        time dependencies of the mixed moments

        :param k_vec: :math:`k` in eq. 11
        :param e_counter: :math:`e` in eq. 11
        :return: :math:`\frac{d\beta}{dt}`
        """

        if len(e_counter) == 0:
            return sp.Matrix(1, len(self.__n_counter), lambda i, j: 0)

        # compute F(x) for EACH REACTION and EACH entry in the EKCOUNTER (eq. 12)
        f_of_x_vec = [self._make_f_of_x(k_vec, ek.n_vector, reac) for (reac, ek)
                      in itertools.product(self.__propensities, e_counter)]

        # compute <F> from f(x) (eq. 12). The result is a list in which each element is a
        # vector in which each element relates to an entry of counter
        f_expectation_vec = [self._make_f_expectation(f) for f in f_of_x_vec]

        # compute s^e for EACH REACTION and EACH entry in the EKCOUNTER . this is a list of scalars
        s_pow_e_vec = sp.Matrix([self._make_s_pow_e(reac_idx, ek.n_vector) for (reac_idx, ek)
                                 in itertools.product(range(len(self.__propensities)), e_counter)])

        # compute (k choose e) for EACH REACTION and EACH entry in the EKCOUNTER . This is a list of scalars.
        # Note that this does not depend on the reaction, so we can just repeat the result for each reaction
        k_choose_e_vec = sp.Matrix(
                [make_k_chose_e(ek.n_vector, k_vec) for ek in e_counter] *
                len(self.__propensities)
                )

        # compute the element-wise product of the three entities
        s_times_ke = s_pow_e_vec.multiply_elementwise(k_choose_e_vec)
        prod = [list(f * s_ke) for (f, s_ke) in zip(f_expectation_vec, s_times_ke)]

        # we have a list of vectors and we want to obtain a list of sums of all nth element together.
        # To do that we put all the data into a matrix in which each row is a different vector
        to_sum = sp.Matrix(prod)

        # then we sum over the columns -> row vector
        mixed_moments = sum_of_cols(to_sum)

        return mixed_moments


    def _make_f_of_x(self, k_vec, e_vec, reaction):
        r"""
        Calculates :math:`F():math:` in eq. 12 (see Ale et al. 2013) for a specific reaction , :math:`k` and :math:`e`

        :param k_vec: the vector :math:`k`
        :param e_vec: the vector :math:`e`
        :param reaction: the equation of the reaction {:math:`a(x) in the model}
        :return: :math:`F()`
        """

        # product of all values of {x ^ (k - e)} for all combination of e and k
        prod = product([var ** (k_vec[i] - e_vec[i]) for i,var in enumerate(self.__species)])
        # multiply the product by the propensity {a(x)}
        return prod * reaction

    @cache
    def _make_f_expectation(self, expr):
        """
        Calculates :math:`<F>` in eq. 12 (see Ale et al. 2013) to calculate :math:`<F>` for EACH VARIABLE combination.

        :param expr: an expression
        :return: a column vector. Each row correspond to an element of counter.
        :rtype: :class:`sympy.Matrix`
        """
        # compute derivatives for EACH ENTRY in COUNTER

        derives = sp.Matrix([derive_expr_from_counter_entry(expr, self.__species, tuple(c.n_vector))
                             for c in self.__n_counter])

        # Computes the factorial terms for EACH entry in COUNTER
        factorial_terms = sp.Matrix([get_one_over_n_factorial(tuple(c.n_vector)) for c in self.__n_counter])

        # Element wise product of the two vectors
        te_vector= derives.multiply_elementwise(factorial_terms)

        return te_vector

    def _make_s_pow_e(self, reac_idx, e_vec):
        """
        Compute s^e in equation 11  (see Ale et al. 2013)

        :param reac_idx: the index (that is the column in the stoichiometry matrix)
         of the reaction to consider.
        :type reac_idx: `int`
        :param e_vec: the vector e
        :return: a scalar (s^e)
        """
        return product([self.__stoichoimetry_matrix[i, reac_idx] ** e for i,e in enumerate(e_vec)])
