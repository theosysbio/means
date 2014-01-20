#######################################################################
#  Expresses higher (2+) order raw moments in terms of central moments
#  Returns momvec (list of central moments, ymi) and mom (list of
#  equivalent expressions in terms of raw moments) (see eq. 8)
#######################################################################

from sympy import Matrix, Symbol
from math import factorial


def raw_to_central(nvariables, counter, ymat, mcounter):
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

    :param nvariables: Number of variables in the system
    :param counter: The first list output by fcount - all moments minus the first order moments
    :param ymat:
    :param mcounter: The second list output by fcount - all moments including the first order moments
    :return:
    """

    ncounter = counter[:]
    ncounter.remove(counter[0])  # TODO: They are certainly relying on counter[0] to always be some moment we don't need as it is removed

    mom = []        #create empty list for mom

    # This loop loops through the ::math::`[n_1, ..., n_d]` vectors of the sums in the beginning of the equation
    # i.e. ::math::`\sum_{k1=0}^n_1 ... \sum_{kd=0}^n_d` part of the equation.
    # Note, this is not the sum over k's in that equation, or at least I think its not
    for Tn in range(0, len(ncounter)):  #loop through all n1,...,nd combinations
        nvec = ncounter[Tn]     # nvec is the vector ::math::`[n_1, ... n_d]` in equation 8


        # This whole block below just generates midx, that contains indices of `mcounter` that are lower than
        # or equal to the current nvec
        # where lower than and equal is defined as ::math::`n_i^a \le n_i^b ~ \textrm{for all i}`
        # I assume this is just generating the list of possible k values to satisfy ns in the equation.
        repmat = []
        for i in range(0, len(mcounter)):
            repmat.append(nvec)

        G = Matrix(mcounter)
        H = Matrix(repmat)
        check = G - H

        midx = []                      #midx is list of mvecs <= current nvec
        for i in range(0, check.rows):
            mc = max(check[i, :])
            if mc <= 0:
                midx.append(i)

        ################################################################
        #  Calculate terms in eq. 8 to sum over for each n1,...,nd combination
        #  Each entry in Taylorexp corresponds to a k1,...,kd combination
        ################################################################

        Taylorexp = Matrix(len(midx), 1, lambda i, j: 0)

        for Tm in range(0, len(midx)):  # This just loops over all the mvectors deemed suitable in the previous block
            mvec = mcounter[midx[Tm]]   # Confusing indexing pattern but just does the above.
            # mvec is the vector ::math::`[k_1, ..., k_d]`

            f_2 = 1                                       # f_2 is (n k) binomial term (TODO: Why call it f_2 then, not "binomial_term"???)
            for fi in range(0, len(mvec)):
                f_2 = f_2 * factorial(nvec[fi]) / (factorial(mvec[fi]) * factorial(nvec[fi] - mvec[fi]))

            f_3 = 1                                      # f_3 is (-1)^(n-k) term
            for fi in range(0, len(mvec)):
                f_3 = f_3 * (-1) ** (nvec[fi] - mvec[fi])

            A = (ymat[0]) ** (nvec[0] - mvec[0])             # alpha term: mu^(n-k)
            for nv in range(1, nvariables):
                A = A * (ymat[nv] ** (nvec[nv] - mvec[nv]))  # Equivalent to ::math::`\mu_i^{n_i - k_i}` in the equation, ymat being the mu

            mstr = str(mvec[0])                          # beta term: x^k
            for mm in range(1, len(mvec)):
                mstr = mstr + str(mvec[mm])
            B = Symbol('x' + mstr)                       # For some reason, the x^k term is stored as x_str(k) symbol
                                                         # e.g. x_120 for k = [1,2,0]

            # Join all things up to complete the part right to the sum operators in equation 8
            Taylorexp[Tm] = f_2 * f_3 * (A * B)              #calculate term for k1,....,kd

        # Make sure to sum across the Taylorexp thingies, not only to store them and put them into mom
        mom.append(sum(Taylorexp))     #sum over k1,...,kd terms for particular n1,...,nd

    # This block of code just traces back the values from ncounter that were used to generate mom
    # and then returns them as list of symbols ym_{n_values}
    momvec = []
    for i in range(0, len(ncounter)):
        vec = ncounter[i]
        mstr = str(vec[0])
        for mm in range(1, len(vec)):
            mstr = mstr + str(vec[mm])          #creates string of indices for each n1,...,nd
        momvec.append(Symbol('ym' + mstr))


    return (mom, momvec)
