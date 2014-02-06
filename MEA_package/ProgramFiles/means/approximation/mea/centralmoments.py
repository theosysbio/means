import operator

import sympy as sp

from means.approximation.mea.eq_mixedmoments import make_k_chose_e, eq_mixedmoments


def all_higher_or_eq(vec_a, vec_b):
    return all([a >= b for a, b in zip(vec_a, vec_b)])

def eq_centralmoments(counter, mcounter, M, ymat, amat, S):
    """
    Function used to calculate the terms required for use in equations giving the time dependence of central moments
    (this is the Equation 9 in the paper).

    :param counter: see `fcount` function
    :param mcounter: see `fcount` function
    :param M: du/dt in paper
    :param ymat: species matrix: y0, y1,..., yd
    :param amat: propensities
    :param S: stoichiometry matrix
    :return: central_moments matrix with `(len(counter)-1)` rows and one column per entry in counter
            This list contains sum of the terms `n_choose_k*minus_one_pow_n_minus_k*(AdB/dt + B dA/dt)` in eq. 9 for each
             n1,...,nd combination in eq. 9 where ... is ... #todo
    """
    central_moments = []

    ###############################################################
    # Loops through required combinations of moments (n1,...,nd)
    # (does not include 0th order central moment as this is 1,
    # or 1st order central moment as this is 0

    # copy M matrix as a list of rows vectors (1/species)
    m_mat = [M[nv, :] for nv in range(M.rows)]
    #todo : tolist()

    for count in counter:
        # skip zeroth moment
        if count.order == 0:
            continue
        nvec = count.n_vector

        # Find all moments in mcounter that are smaller than `nvec`.
        midx = [i for i,c in enumerate(mcounter) if all_higher_or_eq(nvec, c.n_vector)]

        Taylorexp = [[0] * len(counter)] * len(mcounter)

        for (Tm, midx_val) in enumerate(midx):
            mc = mcounter[midx_val]
            mvec = mcounter[midx_val].n_vector

            # (n k) binomial term in equation 9
            n_choose_k = make_k_chose_e(mvec, nvec)

            # (-1)^(n-k) term in equation 9
            minus_one_pow_n_minus_k = reduce(operator.mul, [sp.Integer(-1) ** (n - m) for (n,m) in zip(nvec, mvec)])

            ##########################################
            # Calculate A, dAdt terms in equation 9
            # (equivalent to fA_counter in Angelique's code)
            # ymat used in place of means - these will be replaced later
            A = reduce(operator.mul,  [y ** (n - m) for y,n,m in zip(ymat, nvec, mvec)])
            dAdt = reduce(operator.add, [(n - m) * (y ** (-1)) * A * vec for y,n,m,vec in zip(ymat, nvec, mvec, m_mat)])

            ekcounter = [c for c in mcounter if all_higher_or_eq(mvec, c.n_vector) if c.order > 0]

            dBdt = eq_mixedmoments(amat, counter, S, ymat, mvec, ekcounter)

            if len(ekcounter) == 0:
                B = 1
            else:
                # Calculate B, dBdt terms in equation 9
                B = mc.symbol


            Taylorexp[Tm] = (n_choose_k * minus_one_pow_n_minus_k * (A * dBdt + B * dAdt))

        #################################################
        # Create list of central moments terms to return
        # to MEA program
        # Taylorexp is a matrix which has an entry equal to
        # the f2*f3*(AdB/dt + B dA/dt) term in equation 9
        # for each k1,..,kd
        # These are summed over to give the Taylor Expansion
        # for each n1,..,nd combination in equation 9
        #################################################
        Taylorexp1 = sp.Matrix(Taylorexp)

        centralmomentsTn = [sum(Taylorexp1[:, j]) for j in range(len(counter))]
        #todo row = sph.sum_of_cols(Taylorexp1)

        central_moments.append(centralmomentsTn)



    return sp.Matrix(central_moments)
