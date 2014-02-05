import operator

import sympy as sp

from means.util.sympyhelpers import substitute_all


def gamma_factorial(expr, n):
    if n == 0:
        return 1
    
    return reduce(operator.mul,[ expr+i for i in range(n)])

def gamma_closure(counter, n_species, n_moment, central_from_raw_exprs, central_moments_symbols, gamma_type=1):
    idx_of_last_ord_mom = [i for i,c in enumerate(counter) if sum(c) >= n_moment]
    Y_vec = sp.Matrix([sp.Symbol('Y_{0}'.format(str(i))) for i in range(n_species + 1)])
    #alpha_vec = [sp.Symbol('alpha_' + str(i)) for i in range(n_species + 1)]

    if gamma_type == 1:
        beta_in_matrix = sp.Matrix([Y_vec[0]] + [Y + Y_vec[0] for Y in Y_vec[1:]])
    elif gamma_type == 2:
        beta_in_matrix = sp.Matrix([sum(Y_vec[0:i+1]) for i in range(n_species + 1)])
    else:
        beta_in_matrix = sp.Matrix(Y_vec[1:])

    expectation_symbols = sp.Matrix([sp.Symbol('y_{0}'.format(str(i))) for i in range(n_species)])
    variance_symbols = sp.Matrix([sp.Symbol('ym_{0}'.format(str(i))) for i in range(n_species)])
    beta_exprs = sp.Matrix([v / e for e,v in zip(expectation_symbols,variance_symbols)])
    alpha_bar_exprs = sp.Matrix([(e ** 2) / v for e,v in zip(expectation_symbols,variance_symbols)])

    if gamma_type == 1 or 0:
        first = sp.Matrix([gamma_type])
        alpha_exprs = alpha_bar_exprs - sp.Matrix([gamma_type]*n_species)
        alpha_exprs = first.col_join(alpha_exprs)

    elif gamma_type == 2:
            alpha_exprs_0 = sp.Matrix([1] + [alpha_bar_exprs[0] - 1])
            alpha_exprs = sp.Matrix(alpha_bar_exprs[1:]) - sp.Matrix(alpha_bar_exprs[0:len(alpha_bar_exprs)-1])
            alpha_exprs = alpha_exprs_0.col_join(alpha_exprs)

    alpha_multipliers = []
    beta_multipliers = []
    for row in counter:
        if sum(row) == 0:
            continue
        alpha_multipliers.append(reduce(operator.mul, [(a ** r).expand() for a,r in zip(beta_in_matrix[1:],row)]))
        beta_multipliers.append(reduce(operator.mul, [(b ** r).expand() for b,r in zip(beta_exprs,row)]))

    alpha_multipliers = sp.Matrix(alpha_multipliers)

    ## get alpha-expressions
    for i,a in enumerate(alpha_exprs):
        Y_to_substitute = [sp.Symbol("Y_{0}".format(i))**n for n in range(2, n_moment + 1)]
        alpha_m = [gamma_factorial(a,n) for n in range(2, n_moment +1)]

        subs_pairs = zip(Y_to_substitute, alpha_m)
        subs_pairs.append((sp.Symbol("Y_{0}".format(i)),a ))
        alpha_multipliers = alpha_multipliers.applyfunc(lambda x : substitute_all(x, subs_pairs))

    xs = ["x_" + "_".join(str(r) for r in c) for c in counter if sum(row) != 0]
    gamma_x = sp.Matrix([a *b for a,b in zip(alpha_multipliers, beta_multipliers)])
    subs_pairs = zip(xs,gamma_x)
    gamma_moments = central_from_raw_exprs.applyfunc(lambda x : substitute_all(x, subs_pairs))
    gamma_closed_moments = sp.Matrix([gamma_moments[i-1] for i in idx_of_last_ord_mom])
    closed_moments_symbols = sp.Matrix([central_moments_symbols[i-1] for i in idx_of_last_ord_mom])

    return (gamma_closed_moments, closed_moments_symbols)

