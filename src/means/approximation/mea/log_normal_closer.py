import sympy as sp
from means.util.sympyhelpers import substitute_all

class LogNormalCloser(object):
    def __init__(self):
    close

def get_covariance_symbol(q_counter, sp1_idx, sp2_idx):
    if sp1_idx == sp2_idx:
        return [q.symbol for q in q_counter if q.n_vector[sp1_idx] == 2 and q.order == 2][0]
    return [q.symbol for q in q_counter if q.n_vector[sp1_idx] == 1 and q.n_vector[sp2_idx] == 1 and q.order == 2][0]

def get_log_covariance(log_variance_mat, log_expectation_symbols, covariance_matrix, x, y):

    if x == y:
        return log_variance_mat[x,x]
    else:
        denom = sp.exp(log_expectation_symbols[x] +
                       log_expectation_symbols[y] +
                       (log_variance_mat[x,x] + log_variance_mat[y, y])/ sp.Integer(2))
        return sp.log(sp.Integer(1) + covariance_matrix[x, y] / denom)

def log_normal_closer(n_species, problem_moments):

    covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: get_covariance_symbol(problem_moments,x,y))
    variance_symbols = [covariance_matrix[i, i] for i in range(n_species)]

    expectation_symbols = [pm.symbol for pm in problem_moments if pm.order == 1]
    print "log_variance_mat"
    print expectation_symbols

    log_variance_symbols = sp.Matrix([sp.log(sp.Integer(1) + v/(e ** sp.Integer(2))) for e,v in zip(expectation_symbols, variance_symbols)])
    print "log_variance_symbols"
    print log_variance_symbols

    log_expectation_symbols = sp.Matrix([sp.log(e) - lv/sp.Integer(2) for e,lv in zip(expectation_symbols, log_variance_symbols)])
    print "log_expectation_symbols"
    print log_expectation_symbols

    log_variance_mat = sp.Matrix(n_species,n_species, lambda x,y: log_variance_symbols[x] if x == y else 0)

    print "log_variance_mat"
    print log_variance_mat

    log_covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: \
            get_log_covariance(log_variance_mat, log_expectation_symbols, covariance_matrix, x, y))
    print "log_covariance_matrix"
    print log_covariance_matrix

    pm_n_vecs = [sp.Matrix(pm.n_vector) for pm in problem_moments if pm.order > 1 ]
    print "pm_n_vecs"
    print pm_n_vecs

    out_mat = sp.Matrix([n * (log_covariance_matrix * n.T) / sp.Integer(2) + n * log_expectation_symbols for n in pm_n_vecs])
    out_mat = out_mat.applyfunc(lambda x: sp.expand(sp.exp(x)))

    return out_mat


def log_normal_closer_wrapper(mass_fluctuation_kinetics, prob_moments, central_from_raw_exprs, n_moments, species, k_counter):
    n_species = len(species)

    # as many as as moment of order > 0
    log_normal_closed_raw_moms = log_normal_closer(n_species, prob_moments)

    print "central_from_raw_exprs"
    print central_from_raw_exprs
    print "log_normal_closed_central_moms"
    print log_normal_closed_raw_moms

    # this should have as many symbols as elements in central_from_raw_exprs
    high_order_raw_moments_symb = [k.symbol for k in k_counter if k.order > 1]
    high_order_central_moments = [n for n in prob_moments if n.order > 1]

    substitution_pairs = zip(high_order_raw_moments_symb, log_normal_closed_raw_moms)
    for i in substitution_pairs:
        print i

    closed_central_moments = central_from_raw_exprs.applyfunc(lambda x: substitute_all(x, substitution_pairs))


    print (len(closed_central_moments), len(high_order_central_moments))

    central_substitutions_pairs = [(n.symbol, ccm) for n,ccm in zip(high_order_central_moments, closed_central_moments) if n.order == n_moments]

    for i in central_substitutions_pairs:
        print (i[0], i[1].atoms())

    new_mkf = sp.Matrix([mfk for mfk, pm in zip(mass_fluctuation_kinetics, prob_moments) if pm.order < n_moments])
    new_prob_moments = [pm for pm in prob_moments if pm.order < n_moments]
    new_mkf = new_mkf.applyfunc(lambda x: substitute_all(x, central_substitutions_pairs))


    print "new_mkf"
    print new_mkf

    return new_mkf,new_prob_moments

