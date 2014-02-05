#######################################################################
#  Expresses higher (2+) order raw moments in terms of central moments
#  Returns momvec (list of central moments, ymi) and mom (list of
#  equivalent expressions in terms of raw moments) (see eq. 8)
#######################################################################

import sympy as sp
from centralmoments import all_higher_or_eq
from eq_mixedmoments import make_k_chose_e
import operator

def make_beta(k_vec):
    return sp.Symbol('x_' + "_".join([str(k) for k in k_vec]))

def make_alpha(n_vec, k_vec, ymat):
    return reduce(operator.mul,  [y ** (n - m) for y,n,m in zip(ymat, n_vec, k_vec)])

def make_min_one_pow_n_minus_k(n_vec, k_vec):
    return reduce(operator.mul, [(-1) ** (n - k) for (n,k) in zip(n_vec, k_vec)])

def raw_to_central(n_counter, species, k_counter):
    #todo, fix docstring
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

    :param n_counter: The first list output by fcount - all moments minus the first order moments
    :param species: the list of species/variables
    :param k_counter: The second list output by fcount - all moments including the first order moments
    :return:
    """

    # This block of code just traces back the values from counter that were used to generate mom
    # and then returns them as list of symbols ym_{n_values}
    #momvec = [sp.Symbol("ym_" + "_".join([str(i) for i in c.n_vector])) for c in counter if c.order != 0]

    mom = []        #create empty list for mom


    # This loop loops through the ::math::`[n_1, ..., n_d]` vectors of the sums in the beginning of the equation
    # i.e. ::math::`\sum_{k1=0}^n_1 ... \sum_{kd=0}^n_d` part of the equation.
    # Note, this is not the sum over k's in that equation, or at least I think its not
    for n_iter in n_counter:  #loop through all n1,...,nd combinations

        if n_iter.order == 0:
            continue

        # n_vec is the vector ::math::`[n_1, ... n_d]` in equation 8
        n_vec = n_iter.n_vector

        # k_lower contains the elements of `k_counter` that are lower than or equal to the current n_vec
        # This generates the list of possible k values to satisfy ns in the equation.
        k_lower = [k for k in k_counter if n_iter >= k]

        # k_vec bellow is the vector ::math::`[k_1, ..., k_d]`

        # (n k) binomial term in equation 9
        n_choose_k_vec = [make_k_chose_e(k_vec.n_vector, n_vec) for k_vec in k_lower]

        # (-1)^(n-k) term in equation 9
        minus_one_pow_n_min_k_vec = [make_min_one_pow_n_minus_k(n_vec, k_vec.n_vector)  for k_vec in k_lower ]

        # alpha term in equation 9
        alpha_vec = [make_alpha(n_vec, k_vec.n_vector, species) for k_vec in k_lower]

        # beta term in equation 9
        beta_vec = [k_vec.symbol for k_vec in k_lower]
        # let us multiply all terms
        product = [(n * m * a * b) for (n, m, a, b) in zip(n_choose_k_vec, minus_one_pow_n_min_k_vec, alpha_vec, beta_vec)]

        # and store the product
        mom.append(sum(product))


    return sp.Matrix(mom)
