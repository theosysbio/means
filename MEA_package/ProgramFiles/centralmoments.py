import sympy as sp
import operator
from eq_mixedmoments import eq_mixedmoments
from eq_mixedmoments import make_k_chose_e
from sympyhelpers import sum_of_cols, product, sympy_sum_list

#todo del
def all_higher_or_eq(vec_a, vec_b):
    return all([a >= b for a, b in zip(vec_a, vec_b)])

def eq_centralmoments(n_counter, k_counter, dmu_over_dt, species, propensities, stoichiometry_matrix):
    """
    Function used to calculate the terms required for use in equations giving the time dependence of central moments
    (this is the Equation 9 in the paper).

    :param n_counter: see `fcount` function
    :param k_counter: see `fcount` function
    :param dmu_over_dt: du/dt in paper
    :param species: species matrix: y_0, y_1,..., y_d
    :param propensities: propensities matrix
    :param stoichiometry_matrix: stoichiometry matrix
    :return: central_moments matrix with `(len(counter)-1)` rows and one column per entry in counter
            This list contains sum of the terms `n_choose_k*minus_one_pow_n_minus_k*(AdB/dt + beta dA/dt)` in eq. 9 for each
             n1,...,nd combination in eq. 9 where ... is ... #todo
    """
    central_moments = []


    # Loops through required combinations of moments (n1,...,nd)
    # (does not include 0th order central moment as this is 1,
    # or 1st order central moment as this is 0

    # copy M matrix as a list of rows vectors (1/species)
    dmu_mat = [sp.Matrix(l).T for l in dmu_over_dt.tolist()]
    #todo : tolist()

    for n_iter in n_counter:
        # skip zeroth moment
        if n_iter.order == 0:
            continue
        n_vec = n_iter.n_vector

        # Find all moments in k_counter that are lower than the current n_iter
        k_lower = [k for k in k_counter if n_iter >= k]

        taylor_exp_mat = []

        for k_iter in k_lower:
            k_vec = k_iter.n_vector

            # (n k) binomial term in equation 9
            n_choose_k = make_k_chose_e(k_vec, n_vec)

            # (-1)^(n-k) term in equation 9
            minus_one_pow_n_minus_k = product([sp.Integer(-1) ** (n - m) for (n,m) in zip(n_vec, k_vec)])

            # Calculate alpha, dalpha_over_dt terms in equation 9
            alpha = product([s ** (n - k) for s,n,k in zip(species, n_vec, k_vec)])
            # eq 10 {(n - k) mu_i^(-1)} corresponds to {(n - k)/s}. s is symbol for mean of a species

            # multiplies by alpha an the ith row of dmu_mat and sum it to get dalpha_over_dt
            # eq 10 {(n - k) mu_i^(-1)} corresponds to {(n - k)/s}
            dalpha_over_dt = sympy_sum_list([((n - k) / s) * alpha * mu_row for s,n,k,mu_row in zip(species, n_vec, k_vec, dmu_mat)])

            # e_counter contains elements of k_counter lower than the current k_iter
            e_counter = [k for k in k_counter if k_iter >= k and k.order > 0]

            dbeta_over_dt = eq_mixedmoments(propensities, n_counter, stoichiometry_matrix, species, k_iter, e_counter)

            # Calculate beta, dbeta_over_dt terms in equation 9
            if len(e_counter) == 0:
                beta = 1
            else:
                beta = k_iter.symbol

            taylor_exp_mat.append(n_choose_k * minus_one_pow_n_minus_k * (alpha * dbeta_over_dt + beta * dalpha_over_dt))

        # Taylorexp is a matrix which has an entry equal to
        # the `n_choose_k * minus_one_pow_n_minus_k * (AdB/dt + beta dA/dt)` term in equation 9  for each k1,..,kd
        # These are summed over to give the Taylor Expansion for each n1,..,nd combination in equation 9
        central_moments.append(sum_of_cols(sp.Matrix(taylor_exp_mat)))

    return sp.Matrix(central_moments)
