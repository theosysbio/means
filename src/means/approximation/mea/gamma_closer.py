import sympy as sp
import operator
from zero_closer import CloserBase
from means.util.sympyhelpers import substitute_all

class GammaCloser(CloserBase):
    def __init__(self, max_order, type=1):
        super(GammaCloser, self).__init__(max_order, multivariate = (type > 0))
        self.__type = type

    @property
    def type(self):
        return self.__type


    def get_parameter_symbols(self, prob_moments):
        '''
        Calculates parameters Y expressions and beta coefficients in
        :math: `X = {A(\beta_0,\beta_1\ldots \beta_n) \cdot Y}`


        :param prob_moments: the moments with symbols and moment vectors
        :return: two column matrices Y expressions and beta multipliers
        '''

        gamma_type = self.type
        n_moment = self.max_order
        print "=============================================="
        print "prob_moments"
        print prob_moments
        print "n_moment"
        print n_moment
        n_species = len([None for pm in prob_moments if pm.order == 1])
        # Create symbolic species :math: `Y_0 \sim {Y_n}`, where n is n_species
        symbolic_species = sp.Matrix([sp.Symbol('Y_{0}'.format(str(i))) for i in range(n_species + 1)])


        # Obtain beta terms in multivariate Gamma matrix. See Eq. 1a & 1b in Lakatos 2014 unpublished
        if gamma_type == 1:
            beta_in_matrix = sp.Matrix([Y + symbolic_species[0] for Y in symbolic_species[1:]])
        elif gamma_type == 2:
            beta_in_matrix = sp.Matrix([sum(symbolic_species[0:i+1]) for i in range(n_species + 1)])
        else:
            beta_in_matrix = sp.Matrix(symbolic_species[1:])

        # E() and Var() symbols for each species have already been made in prob_moments matrix
        expectation_symbols = sp.Matrix([n.symbol for n in prob_moments if n.order == 1])
        variance_symbols = []
        for sp_idx in range(n_species):
            variance_symbols += [p.symbol for p in prob_moments if p.n_vector[sp_idx] == 2 and p.order == 2]
        variance_symbols = sp.Matrix(variance_symbols)

        # Compute :math:  `\beta_i = Var(X_i)/\mathbb{E}(X_i) \bar\alpha_i = \mathbb{E}(X_i)^2/Var(X_i)`
        beta_exprs = sp.Matrix([v / e for e,v in zip(expectation_symbols,variance_symbols)])
        alpha_bar_exprs = sp.Matrix([(e ** 2) / v for e,v in zip(expectation_symbols,variance_symbols)])

        # Gamma type 1 :math: `\bar\alpha_i = \alpha_0 + \alpha_i`
        # Gamma type 1: covariance is :math: `\alpha_0 * \beta_i * \beta_j`, so :math: `\alpha_0` should be calculated
        # but as it will force :math: `\alpha_0` to be negative
        # resulting ODEs are not solvable, so set arbitrary :math: `\alpha_0`
        # Arbitrary value 1 here is adopted from MATLAB code.
        # Gamma type 0 (univariate case): :math: `alpha_0 =0`
        # Thus :math: `alpha_0` for Gamma type 0 and 1 happen to be the same as the gamma_type
        if gamma_type != 2:
            first = sp.Matrix([gamma_type])
            alpha_exprs = alpha_bar_exprs - sp.Matrix([gamma_type]*n_species)
            alpha_exprs = first.col_join(alpha_exprs)

        # Gamma type 2 has arbitrary alpha0 fixme why is it arbitrary
        # Gamma type 2 :math: `\bar\alpha_i = \sum \limits_{i}  \alpha_i`
        else: # if gamma_type == 2:
            first = sp.Matrix([1] + [alpha_bar_exprs[0] - 1])
            alpha_exprs = sp.Matrix(alpha_bar_exprs[1:]) - sp.Matrix(alpha_bar_exprs[0:len(alpha_bar_exprs)-1])
            alpha_exprs = first.col_join(alpha_exprs)

        # Each row in moment matrix contains the exponents of Xs for a given moment
        # Each row in Y_exprs and beta_multipliers has elements on the appropriate power
        # determined by the corresponding row in the moment matrix
        Y_exprs = []
        beta_multipliers = []
        for mom in prob_moments:
            if mom.order < 2:
                continue
            Y_exprs.append(reduce(operator.mul, [(b ** s).expand() for b, s in zip(beta_in_matrix, mom.n_vector)]))
            beta_multipliers.append(reduce(operator.mul, [(b ** s).expand() for b, s in zip(beta_exprs, mom.n_vector)]))


        Y_exprs = sp.Matrix(Y_exprs)
        beta_multipliers = sp.Matrix(beta_multipliers)
        Y_exprs = Y_exprs.applyfunc(sp.expand)

        # Substitute alpha expressions in place of symbolic species Ys
        # by going through all powers up to the moment order for closure
        subs_pairs = []
        for i,a in enumerate(alpha_exprs):
            Y_to_substitute = [sp.Symbol("Y_{0}".format(i))**n for n in range(2, n_moment+1)]

            # Obtain alpha term for higher older moments :math: `\mathbb{E}(X_i^m) = (\bar\alpha_i)_m\beta_i^m`
            alpha_m = [self.gamma_factorial(a,n) for n in range(2, n_moment+1)]

            # Substitute alpha term for symbolic species
            subs_pairs += zip(Y_to_substitute, alpha_m)
            subs_pairs.append((sp.Symbol("Y_{0}".format(i)), a))
        Y_exprs = substitute_all(Y_exprs, subs_pairs)

        return Y_exprs, beta_multipliers

    def compute_raw_moments(self, n_counter, k_counter):
        '''
        Compute :math: `X_i`
        Gamma type 1: :math: `X_i = \frac {\beta_i}{\beta_0}Y_0 + Y_i`
        Gamma type 2: :math: `X_i = \sum_{k=0}^{i}  \frac {\beta_i}{\beta_k}Y_k`

        :param problem_moments: moment matrix with central moment symbols
        :return:
        '''


        prob_moments_over_dt = [k for k in k_counter if k.order == 1]
        # and the higher order central moments (variances, covariances,...)
        prob_moments_over_dt += [n for n in n_counter if n.order > 1]


        alpha_multipliers, beta_multipliers = self.get_parameter_symbols(prob_moments_over_dt)

        print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        for i in alpha_multipliers:
            print i
        print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        out_mat = sp.Matrix([a * b for a,b in zip(alpha_multipliers, beta_multipliers)])
        out_mat = out_mat.applyfunc(sp.expand)
        return out_mat


    def gamma_factorial(self, expr, n):
        '''
        Compute :math: `\frac {(\alpha)_m = (\alpha + m - 1)!}{(\alpha - 1)!}`
        See Eq. 3 in Gamma moment closure Lakatos 2014 unpublished

        :param expr:
        :param n:
        :return:
        '''
        if n == 0:
            return 1
        return reduce(operator.mul, [expr+i for i in range(n)])
