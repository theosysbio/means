#####################################################################
# Called by centralmoments.py
# Provides the terms needed for equation 11 in Angelique's paper
# This gives the expressions for dB/dt in equation 9, these are the 
# time dependencies of the mixed moments
####################################################################
import itertools
from means.util.sympyhelpers import sum_of_cols, product
import sympy as sp
from means.approximation.mea.TaylorExpansion import derive_expr_from_counter_entry
from means.approximation.mea.TaylorExpansion import get_factorial_term
from means.util.decorators import cache

def make_k_chose_e(e_vec, k_vec):
    """
    Computes the product k chose e

    :param e_vec: the vector e
    :param k_vec: the vector k
    :return: a scalar
    """
    return  product([sp.factorial(k) / (sp.factorial(e) * sp.factorial(k - e)) for e,k in zip(e_vec, k_vec)])

class DBetaOverDtCalculator(object):
    def __init__(self, propensities, n_counter, stoichoimetry_matrix, species):
        self.__propensities = propensities
        self.__n_counter = n_counter
        self.__stoichoimetry_matrix = stoichoimetry_matrix
        self.__species = tuple(species)

    def get(self, k_iter, e_counter):
        r"""
        Provides the terms needed for equation 11 (see Ale et al. 2013).
        This gives the expressions for :math:`\frac{d\beta}{dt}` in equation 9, these are the
        time dependencies of the mixed moments

        :param propensities:    propensities
        :param n_counter: a list of all possible combination of order of derivation
        :param S: The stoichiometry matrix. Explicitly provided by the model
        :param species: the names of the variables/species
        :param k_vec: k in eq. 11
        :param e_counter: e in eq. 11

        :return: :math:`\frac{d\beta}{dt}`
        """

        if len(e_counter) == 0:
            return sp.Matrix(1, len(self.__n_counter), lambda i, j: 0)

        # compute F(x) for EACH REACTION and EACH entry in the EKCOUNTER (eq. 12)
        f_of_x_vec = [self.make_f_of_x(k_iter.n_vector, ek.n_vector, reac) for (reac, ek) in
                      itertools.product(self.__propensities, e_counter)]

        # compute <F> from f(x) (eq. 12). The result is a list in which each element is a
        # vector in which each element relates to an entry of counter

        f_expectation_vec = [self.make_f_expectation(f) for f in f_of_x_vec]

        # compute s^e for EACH REACTION and EACH entry in the EKCOUNTER . this is a list of scalars
        s_pow_e_vec = [self.make_s_pow_e(reac_idx, ek.n_vector) for (reac_idx, ek) in
                       itertools.product(range(len(self.__propensities)), e_counter)]

        # compute (k choose e) for EACH REACTION and EACH entry in the EKCOUNTER . This is a list of scalars.
        # Note that this does not depend on the reaction, so we can just repeat the result for each reaction
        k_choose_e_vec = [make_k_chose_e(ek.n_vector, k_iter.n_vector)
                          for ek in e_counter] * len(self.__propensities)

        # compute the element-wise product of the three entities
        prod = [f * s * ke for (f, s, ke) in zip(f_expectation_vec, s_pow_e_vec, k_choose_e_vec)]

        # we have a list of vectors and we want to obtain a list of sums of all nth element together.
        # To do that we put all the data into a matrix in which each row is a different vector
        to_sum = sp.Matrix(prod).reshape(len(prod),len(prod[0]))

        # then we sum over the columns -> row vector
        mixed_moments = sum_of_cols(to_sum)

        return mixed_moments


    def make_f_of_x(self, k_vec, e_vec, reaction):
        r"""
        Calculates F() in eq. 12 (see Ale et al. 2013) for a specific reaction , k and e

        :param k_vec: the vector k
        :param e_vec: the vector e
        :param reaction: the equation of the reaction {a(x) in the model}
        :return:
        """

        # product of all values of {x ^ (k - e)} for all combination of e and k
        prod = product([var ** (k_vec[i] - e_vec[i]) for i,var in enumerate(self.__species)])
        # multiply the product by the propensity {a(x)}
        return prod * reaction

    @cache
    def make_f_expectation(self, expr):
        """
        Calculates <F> in eq. 12 (see Ale et al. 2013) to calculate <F> for EACH VARIABLE combination.

        :param expr: an expression
        :param counter: a list of all possible combination of order of derivation
        :return: a column vector (as a sympy matrix). Each row correspond to an element of counter
        """
        # compute derivatives for EACH ENTRY in COUNTER

        derives = [derive_expr_from_counter_entry(expr, self.__species, tuple(c.n_vector)) for c in self.__n_counter]

        # Computes the factorial terms for EACH entry in COUNTER
        factorial_terms = [get_factorial_term(tuple(c.n_vector)) for c in self.__n_counter]

        # Element wise product of the two vectors
        te_matrix = sp.Matrix(len(self.__n_counter), 1, [d*f for (d, f) in zip(derives, factorial_terms)])

        return te_matrix
        # return [d*f for (d, f) in zip(derives, factorial_terms)]


    def make_s_pow_e(self, reac_idx, e_vec):
        """
        Compute s^e in equation 11  (see Ale et al. 2013)

        :param reac_idx: the index of the reaction to consider
        :param e_vec: the vector e
        :return: a scalar (s^e)
        """
        return product([self.__stoichoimetry_matrix[i, reac_idx] ** e for i,e in enumerate(e_vec)])
