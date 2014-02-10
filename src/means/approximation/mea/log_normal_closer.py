import sympy as sp

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

def log_normal_closer(n_species, n_counter, k_counter):

    covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: get_covariance_symbol(n_counter,x,y))
    variance_symbols = [covariance_matrix[i, i] for i in range(n_species)]

    expectation_symbols = [k.symbol for k in k_counter if k.order == 1]


    log_variance_symbols = sp.Matrix([sp.log(sp.Integer(1) + v/(e ** sp.Integer(2))) for e,v in zip(expectation_symbols, variance_symbols)])

    log_expectation_symbols = sp.Matrix([sp.log(e) - lv/sp.Integer(2) for e,lv in zip(expectation_symbols, log_variance_symbols)])

    log_variance_mat = sp.Matrix(n_species,n_species, lambda x,y: log_variance_symbols[x] if x == y else 0)


    log_covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: \
            get_log_covariance(log_variance_mat, log_expectation_symbols, covariance_matrix, x, y))


    n_counter_vecs = [sp.Matrix(n.n_vector) for n in n_counter if n.order >0 ]
    # [sp.Matrix(n.n_vector) * log_expectation_symbols for n in n_counter if n.order >0 ]

    out_mat = sp.Matrix([n * (log_covariance_matrix * n.T) / sp.Integer(2) + n * log_expectation_symbols for n in n_counter_vecs])
    out_mat = out_mat.applyfunc(lambda x: sp.expand(sp.exp(x)))

    return out_mat

def log_normal_closer_wrapper(central_from_raw_exprs, n_counter, k_counter, n_moments, species):

    central_symbols = [n.symbol for n in n_counter if n.order > 0]

    n_species = len(species)

    log_normal_closer(n_species, n_counter, k_counter)
    #... todo
    return #... todo

