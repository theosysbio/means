import sympy as sp
import operator
from eq_mixedmoments import eq_mixedmoments
from eq_mixedmoments import make_k_chose_e

def all_higher_or_eq(a, b):
    return all([a >= b for a, b in zip(a, b)])

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
    :return: centralmoments list of size `(len(counter)-1)` containing an entry for each n1,...,nd combination
            (i.e. each value of counter)
            This list contains sum of the terms `f2*f3*(AdB/dt + B dA/dt)` in eq. 9 for each n1,...,nd combination in eq. 9
    """
    centralmoments = []

    ###############################################################
    # Loops through required combinations of moments (n1,...,nd)
    # (does not include 0th order central moment as this is 1,
    # or 1st order central moment as this is 0


    # copy matrix as a list of rows vectors (1/species)
    m_mat = [M[nv, :] for nv in range(M.rows)]

    for nvec in counter:

        # skip zeroth moment
        if sum(nvec) == 0:
            continue

        # Find all moments in mcounter that are smaller than `nvec`.
        midx = [i for i,c in enumerate(mcounter) if all_higher_or_eq(nvec, c)]

        Taylorexp = [[0] * len(counter)] * len(mcounter)

        for Tm in range(0, len(midx)):
            mvec = mcounter[midx[Tm]]   #equivalent to k in paper

            # (n k) binomial term in equation 9
            n_choose_k = make_k_chose_e(mvec, nvec)

            # (-1)^(n-k) term in equation 9
            minus_one_pow_n_minus_k = reduce(operator.mul, [(-1) ** (n - m) for (n,m) in zip(nvec, mvec)])

            ##########################################
            # Calculate A, dAdt terms in equation 9
            # (equivalent to fA_counter in Angelique's code)
            # ymat used in place of means - these will be replaced later

            A = reduce(operator.mul,  [y ** (n - m) for y,n,m in zip(ymat, nvec, mvec)])

            dAdt = reduce(operator.add, [(n - m) * (y ** (-1)) * A * vec for y,n,m,vec in zip(ymat, nvec, mvec, m_mat)])


            # this is different from before, because it uses mvec i.e. k
            ekcounter = [c for c in mcounter if all_higher_or_eq(mvec, c) if sum(c) > 0]

            dBdt = eq_mixedmoments(amat, counter, S, ymat, mvec, ekcounter)


            if len(ekcounter) == 0:
                B = 1
            else:
                # Calculate B, dBdt terms in equation 9
                B = sp.S("x_" + "_".join([str(s) for s in mvec]))


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

        centralmoments.append(centralmomentsTn)

    return centralmoments
