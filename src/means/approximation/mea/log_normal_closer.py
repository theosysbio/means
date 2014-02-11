import sympy as sp
from means.util.sympyhelpers import substitute_all

# class LogNormalCloser(object):
#     def __init__(self):
#     close

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

def compute_raw_moments(n_species, problem_moments):

    covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: get_covariance_symbol(problem_moments,x,y))
    variance_symbols = [covariance_matrix[i, i] for i in range(n_species)]

    expectation_symbols = [pm.symbol for pm in problem_moments if pm.order == 1]
    log_variance_symbols = sp.Matrix([sp.log(sp.Integer(1) + v/(e ** sp.Integer(2))) for e,v in zip(expectation_symbols, variance_symbols)])
    log_expectation_symbols = sp.Matrix([sp.log(e) - lv/sp.Integer(2) for e,lv in zip(expectation_symbols, log_variance_symbols)])

    log_variance_mat = sp.Matrix(n_species,n_species, lambda x,y: log_variance_symbols[x] if x == y else 0)

    log_covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: \
            get_log_covariance(log_variance_mat, log_expectation_symbols, covariance_matrix, x, y))
    pm_n_vecs = [sp.Matrix(pm.n_vector) for pm in problem_moments if pm.order > 1 ]

    out_mat = sp.Matrix([n * (log_covariance_matrix * n.T) / sp.Integer(2) + n * log_expectation_symbols for n in pm_n_vecs])
    out_mat = out_mat.applyfunc(lambda x: sp.expand(sp.exp(x)))

    return out_mat


def log_normal_closer_wrapper(mass_fluctuation_kinetics, prob_moments, central_from_raw_exprs, n_moments, species, k_counter):
    n_species = len(species)

    # as many as as moment of order > 0
    log_normal_closed_raw_moms = compute_raw_moments(n_species, prob_moments)
    # this should have as many symbols as elements in central_from_raw_exprs
    high_order_raw_moments_symb = [k.symbol for k in k_counter if k.order > 1]
    high_order_central_moments = [n for n in prob_moments if n.order > 1]
    substitution_pairs = zip(high_order_raw_moments_symb, log_normal_closed_raw_moms)
    closed_central_moments = central_from_raw_exprs.applyfunc(lambda x: substitute_all(x, substitution_pairs))

    central_substitutions_pairs = [(n.symbol, ccm) for n,ccm in zip(high_order_central_moments, closed_central_moments) if n.order == n_moments]


    non_null_prob_moment = [pm for pm in prob_moments if pm.order >0]
    new_mkf = sp.Matrix([mfk for mfk, pm in zip(mass_fluctuation_kinetics, non_null_prob_moment) if pm.order < n_moments])
    new_prob_moments = [pm for pm in non_null_prob_moment if pm.order < n_moments]
    new_mkf = new_mkf.applyfunc(lambda x: substitute_all(x, central_substitutions_pairs))


    return new_mkf,new_prob_moments

