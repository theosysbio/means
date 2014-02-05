#######################################################################
#  Expresses higher (2+) order raw moments in terms of central moments
#  Returns momvec (list of central moments, ymi) and mom (list of
#  equivalent expressions in terms of raw moments) (see eq. 8)
#######################################################################

import operator

import sympy as sp

from means.approximation.mea.centralmoments import all_higher_or_eq
from eq_mixedmoments import make_k_chose_e


def make_beta(k_vec):
    return sp.Symbol('x_' + "_".join([str(k) for k in k_vec]))

def make_alpha(n_vec, k_vec, ymat):
    return reduce(operator.mul,  [y ** (n - m) for y,n,m in zip(ymat, n_vec, k_vec)])

def make_min_one_pow_n_minus_k(n_vec, k_vec):
    return reduce(operator.mul, [(-1) ** (n - k) for (n,k) in zip(n_vec, k_vec)])

def raw_to_central(counter, ymat, mcounter):
    """
    Expresses higher (2+) order raw moments in terms of central moments.
    Returns `momvec` (list of central moments, `ymi`) and `mom` (list of equivalent expressions in terms of raw moments).

    Based on equation 8 in the paper:

    ::math::`\mathbf{M_{x^n}} = \sum_{k_1=0}^{n_1} ... \sum_{k_d=0}^{n_d} \mathbf{{n \choose k}} (-1)^{\mathbf{n-k}} \mu^{\mathbf{n-k}} \langle \mathbf{x^k} \rangle`

    The term ::math::`\mathbf{M_{x^n}}` is named `ym{str(n)}` in the output, where `{str(n)}` is string representation
    of vector n.

    The term ::math::`\mu^{\mathbf{n-k}}`, so called alpha term is expressed with respect to `ymat` values that
    are equivalent to ::math::`\mu_i` in the paper.

    The last term, the beta term, ::math::`\langle \mathbf{x^n} \rangle` is named as `xstr(k)` in the resulting
    symbolic expression, where k is the vector of ks (or an element of `mcounter` if you like)

    :param counter: The first list output by fcount - all moments minus the first order moments
    :param ymat: the list of species/variables
    :param mcounter: The second list output by fcount - all moments including the first order moments
    :return:
    """

    # This block of code just traces back the values from counter that were used to generate mom
    # and then returns them as list of symbols ym_{n_values}
    momvec = [sp.Symbol("ym_" + "_".join([str(i) for i in c])) for c in counter if sum(c) != 0]

    mom = []        #create empty list for mom

    # This loop loops through the ::math::`[n_1, ..., n_d]` vectors of the sums in the beginning of the equation
    # i.e. ::math::`\sum_{k1=0}^n_1 ... \sum_{kd=0}^n_d` part of the equation.
    # Note, this is not the sum over k's in that equation, or at least I think its not
    for nvec in counter:  #loop through all n1,...,nd combinations
        # nvec is the vector ::math::`[n_1, ... n_d]` in equation 8
        if sum(nvec) == 0:
            continue

        # m_lower contains the elements of `mcounter` that are lower than or equal to the current nvec
        # where lower than and equal is defined as ::math::`n_i^a \le n_i^b ~ \textrm{for all i}`
        # we assume this is just generating the list of possible k values to satisfy ns in the equation.
        m_lower = [c for c in mcounter if all_higher_or_eq(nvec, c)]

        # mvec is the vector ::math::`[k_1, ..., k_d]`

        # (n k) binomial term in equation 9
        n_choose_k_vec = [make_k_chose_e(mvec, nvec) for mvec in m_lower]

        # (-1)^(n-k) term in equation 9
        minus_one_pow_n_min_k_vec = [make_min_one_pow_n_minus_k(nvec,mvec)  for mvec in m_lower ]

        # alpha term in equation 9
        alpha_vec = [ make_alpha(nvec,mvec, ymat) for mvec in m_lower]

        # beta term in equation 9
        beta_vec = [ make_beta(mvec) for mvec in m_lower]

        # let us multiply all terms
        product = [(n * m * a * b) for (n, m, a, b) in zip(n_choose_k_vec, minus_one_pow_n_min_k_vec, alpha_vec, beta_vec)]

        # and store the product
        mom.append(sum(product))


    return (sp.Matrix(mom), sp.Matrix(momvec))
