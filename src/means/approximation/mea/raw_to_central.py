import operator
import sympy as sp
from means.approximation.mea.mea_helpers import make_k_chose_e


def _make_alpha(n_vec, k_vec, ymat):
    return reduce(operator.mul,  [y ** (n - m) for y, n, m in zip(ymat, n_vec, k_vec)])

def _make_min_one_pow_n_minus_k(n_vec, k_vec):
    return reduce(operator.mul, [(-1) ** (n - k) for n, k in zip(n_vec, k_vec)])

def raw_to_central(n_counter, species, k_counter):
    """
    Expresses central moments in terms of raw moments (and other central moments).
    Based on equation 8 in the paper:

    .. math::
         \mathbf{M_{x^n}} = \sum_{k_1=0}^{n_1} ... \sum_{k_d=0}^{n_d} \mathbf{{n \choose k}} (-1)^{\mathbf{n-k}} \mu^{\mathbf{n-k}} \langle \mathbf{x^k} \\rangle


    The term :math:`\mu^{\mathbf{n-k}}`, so called alpha term is expressed with respect to `species` values that
    are equivalent to :math:`\mu_i` in the paper.

    The last term, the beta term, :math:`\langle \mathbf{x^n} \\rangle` is simply obtained
    from k_counter as it contains the symbols for raw moments.

    :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
    :type n_counter: list[:class:`~means.core.descriptors.Moment`]

    :param species: the symbols for species means

    :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
    :type k_counter: list[:class:`~means.core.descriptors.Moment`]


    :return: a vector of central moments expressed in terms of raw moment
    """
    # create empty output
    central_in_terms_of_raw = []
    # This loop loops through the ::math::`[n_1, ..., n_d]` vectors of the sums in the beginning of the equation
    # i.e. :math:`\sum_{k1=0}^n_1 ... \sum_{kd=0}^n_d` part of the equation.
    # Note, this is not the sum over k's in that equation, or at least I think its not
    for n_iter in n_counter:  #loop through all n1,...,nd combinations
        # nothing to do for 0th order central moment
        if n_iter.order == 0:
            continue
        # n_vec is the vector ::math::`[n_1, ... n_d]` in equation 8
        n_vec = n_iter.n_vector
        # k_lower contains the elements of `k_counter` that are lower than or equal to the current n_vec
        # This generates the list of possible k values to satisfy ns in the equation.
        # `k_vec` iterators bellow are the vector ::math::`[k_1, ..., k_d]`
        k_lower = [k for k in k_counter if n_iter >= k]
        # (n k) binomial term in equation 9
        n_choose_k_vec = [make_k_chose_e(k_vec.n_vector, n_vec) for k_vec in k_lower]
        # (-1)^(n-k) term in equation 9
        minus_one_pow_n_min_k_vec = [_make_min_one_pow_n_minus_k(n_vec, k_vec.n_vector)  for k_vec in k_lower ]
        # alpha term in equation 9
        alpha_vec = [_make_alpha(n_vec, k_vec.n_vector, species) for k_vec in k_lower]
        # beta term in equation 9
        beta_vec = [k_vec.symbol for k_vec in k_lower]
        # let us multiply all terms
        product = [(n * m * a * b) for (n, m, a, b) in zip(n_choose_k_vec, minus_one_pow_n_min_k_vec, alpha_vec, beta_vec)]
        # and store the product
        central_in_terms_of_raw.append(sum(product))
    return sp.Matrix(central_in_terms_of_raw)
