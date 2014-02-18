import sympy as sp
import operator
from means.util.sympyhelpers import substitute_all
from zero_closer import CloserBase


class GammaCloser(CloserBase):
    def __init__(self, n_moments, type=0):
        super(GammaCloser, self).__init__(n_moments)
        self.__is_multivariate = (type > 0)
        self.__type = type

    @property
    def is_multivariate(self):
        return self.__is_multivariate

    @property
    def type(self):
        return self.__type

    def close(self,central_moments_exprs, dmu_over_dt, central_from_raw_exprs, species, n_counter, k_counter):
        mfk = self.generate_mass_fluctuation_kinetics(central_moments_exprs, dmu_over_dt, n_counter)
        prob_lhs = self.generate_problem_left_hand_side(n_counter, k_counter)

        mfk, prob_lhs = self.parametric_closer_wrapper(mfk, central_from_raw_exprs, species, k_counter, prob_lhs)
        return mfk, prob_lhs

    def get_parameter_symbols(self, n_species, prob_moments):
        #fixme set cross terms to zero for gamma type 0

        gamma_type = self.type
        n_moment = self.n_moments

        # Create symbolic species Y0 - Yn, where n = n_species
        symbolic_species = sp.Matrix([sp.Symbol('Y_{0}'.format(str(i))) for i in range(n_species + 1)])


        # Obtain beta terms in the gamma matrix
        if gamma_type == 1:
            beta_in_matrix = sp.Matrix([Y + symbolic_species[0] for Y in symbolic_species[1:]])
        elif gamma_type == 2:
            beta_in_matrix = sp.Matrix([sum(symbolic_species[0:i+1]) for i in range(n_species + 1)])
        else:
            beta_in_matrix = sp.Matrix(symbolic_species[1:])

        expectation_symbols = sp.Matrix([n.symbol for n in prob_moments if n.order == 1])

        variance_symbols = []
        for sp_idx in range(n_species):
            variance_symbols += [p.symbol for p in prob_moments if p.n_vector[sp_idx] == 2 and p.order == 2]
        variance_symbols = sp.Matrix(variance_symbols)
        # Use Eq. 4 to calculate alpha bar and beta
        beta_exprs = sp.Matrix([v / e for e,v in zip(expectation_symbols,variance_symbols)])
        alpha_bar_exprs = sp.Matrix([(e ** 2) / v for e,v in zip(expectation_symbols,variance_symbols)])


        # Gamma type 1: covariance is alpha0 * betai * betaj, so should be taken into account,
        # but as it will force alpha0 to be negative, resulting ODEs are not solvable, so set arbitrary alpha0
        # i.e. alpha_exprs[0] = 1. same as the gamma_type
        # Gamma type 0 (univariate case): covariance is zero, same as the gamma_type too.
        if gamma_type == 1 or 0:
            first = sp.Matrix([gamma_type])
            alpha_exprs = alpha_bar_exprs - sp.Matrix([gamma_type]*n_species)
            alpha_exprs = first.col_join(alpha_exprs)

        # Gamma type 2 has arbitrary alpha0
        else: # if gamma_type == 2:
            first = sp.Matrix([1] + [alpha_bar_exprs[0] - 1])
            alpha_exprs = sp.Matrix(alpha_bar_exprs[1:]) - sp.Matrix(alpha_bar_exprs[0:len(alpha_bar_exprs)-1])
            alpha_exprs = first.col_join(alpha_exprs)

        alpha_multipliers = []
        beta_multipliers = []
        for mom in prob_moments:
            if mom.order < 2:
                continue
            alpha_multipliers.append(reduce(operator.mul, [(b ** s).expand() for b, s in zip(beta_in_matrix, mom.n_vector)]))
            beta_multipliers.append(reduce(operator.mul, [(b ** s).expand() for b, s in zip(beta_exprs, mom.n_vector)]))


        alpha_multipliers = sp.Matrix(alpha_multipliers)
        beta_multipliers = sp.Matrix(beta_multipliers)
        alpha_multipliers = alpha_multipliers.applyfunc(sp.expand)

        ## get alpha-expressions
        subs_pairs = []
        for i,a in enumerate(alpha_exprs):
            Y_to_substitute = [sp.Symbol("Y_{0}".format(i))**n for n in range(2, n_moment+1)]
            alpha_m = [self.gamma_factorial(a,n) for n in range(2, n_moment+1)]

            subs_pairs += zip(Y_to_substitute, alpha_m)
            subs_pairs.append((sp.Symbol("Y_{0}".format(i)), a))
        alpha_multipliers = alpha_multipliers.applyfunc(lambda x: substitute_all(x, subs_pairs))

        return alpha_multipliers, beta_multipliers

    def compute_raw_moments(self, n_species, problem_moments):
        alpha_multipliers, beta_multipliers = self.get_parameter_symbols(n_species, problem_moments)

        out_mat = sp.Matrix([a * b for a,b in zip(alpha_multipliers, beta_multipliers)])
        out_mat = out_mat.applyfunc(sp.expand)
        return out_mat

    def compute_closed_central_moments(self, closed_raw_moments, central_from_raw_exprs, k_counter):
        """
        Replace raw moment terms in central moment expressions by parameters (e.g. mean, variance, covariances)

        :param closed_raw_moments: the expression of all raw moments (expect 0th) in terms of
        parameters such as variance/covariance (i.e. central moments) and first order raw moment (i.e. means)
        :param central_from_raw_exprs: the expression of central moments in terms of raw moments
        :param k_counter: a list of `Moment` object corresponding to raw moment symbols an descriptors
        :return: the central moments where raw moments have been replaced by parametric expressions
        :rtype: sympy.Matrix
        """
        # raw moment lef hand side symbol
        raw_symbols = [raw.symbol for raw in k_counter if raw.order > 1]
        # we want to replace raw moments symbols with closed raw moment expressions (in terms of variances/means)
        substitution_pairs = zip(raw_symbols, closed_raw_moments)

        # for i in substitution_pairs:
        #     print i
        # print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>central_from raw"
        # print central_from_raw_exprs
        # so we can obtain expression of central moments in terms of low order raw moments
        closed_central_moments = substitute_all(central_from_raw_exprs, substitution_pairs)
        return closed_central_moments



    def parametric_closer_wrapper(self, mfk, central_from_raw_exprs, species, k_counter, prob_moments):
        n_moments = self.n_moments
        n_species = len(species)
        # we compute all raw moments according to means / variance/ covariance
        # at this point we have as many raw moments expressions as non-null central moments
        closed_raw_moments = self.compute_raw_moments(n_species, prob_moments)
        # we obtain expressions for central moments in terms of closed raw moments
        closed_central_moments = self.compute_closed_central_moments(closed_raw_moments, central_from_raw_exprs, k_counter)
        # set mixed central moment to zero iff univariate
        #fixme
        #closed_central_moments = self.set_mixed_moments_to_zero(closed_central_moments,prob_moments)

        # we remove ODEs of highest order in mfk
        new_mkf = sp.Matrix([mfk for mfk, pm in zip(mfk, prob_moments) if pm.order < n_moments])
        # new_mkf = mfk
        # retrieve central moments from problem moment. Typically, :math: `[yx2, yx3, ...,yxN]`.
        n_counter = [n for n in prob_moments if n.order > 1]
        # now we want to replace the new mfk (i.e. without highest order moment) any
        # symbol for highest order central moment by the corresponding expression (computed above)
        substitutions_pairs = [(n.symbol, ccm) for n,ccm in zip(n_counter, closed_central_moments) if n.order == n_moments]
        new_mkf = substitute_all(new_mkf, substitutions_pairs)

        # we also update problem moments (aka lhs) to match remaining rhs (aka mkf)
        new_prob_moments = [pm for pm in prob_moments if pm.order < n_moments]
        #new_prob_moments = prob_moments

        return new_mkf,new_prob_moments

    def gamma_factorial(self, expr, n):
        if n == 0:
            return 1
        return reduce(operator.mul,[ expr+i for i in range(n)])
