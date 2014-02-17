import sympy as sp
from means.util.sympyhelpers import substitute_all
from zero_closer import CloserBase
class LogNormalCloser(CloserBase):
    def __init__(self,n_moments, multivariate = True):
        super(LogNormalCloser, self).__init__(n_moments)
        self.__is_multivariate = multivariate
        #fixme implement multivariate
        if not multivariate:
            raise NotImplementedError("todo: implement univariate")

    @property
    def is_multivariate(self):
        return self.__is_multivariate

    def close(self,central_moments_exprs, dmu_over_dt, central_from_raw_exprs, species, n_counter, k_counter):
        mfk = self.generate_mass_fluctuation_kinetics(central_moments_exprs, dmu_over_dt, n_counter)
        prob_lhs = self.generate_problem_left_hand_side(n_counter, k_counter)

        mfk, prob_lhs = self.parametric_closer_wrapper(mfk, central_from_raw_exprs, species, k_counter, prob_lhs)
        return mfk, prob_lhs

    def get_covariance_symbol(self, q_counter, sp1_idx, sp2_idx):
        if sp1_idx == sp2_idx:
            return [q.symbol for q in q_counter if q.n_vector[sp1_idx] == 2 and q.order == 2][0]
        return [q.symbol for q in q_counter if q.n_vector[sp1_idx] == 1 and q.n_vector[sp2_idx] == 1 and q.order == 2][0]

    def get_log_covariance(self, log_variance_mat, log_expectation_symbols, covariance_matrix, x, y):

        if x == y:
            return log_variance_mat[x,x]
        elif self.is_multivariate:
            denom = sp.exp(log_expectation_symbols[x] +
                           log_expectation_symbols[y] +
                           (log_variance_mat[x,x] + log_variance_mat[y, y])/ sp.Integer(2))
            return sp.log(sp.Integer(1) + covariance_matrix[x, y] / denom)
        # univariate case:
        else:
            return sp.Integer(0)

    def compute_raw_moments(self, n_species, problem_moments):
        # The covariance expressed in terms of central moment symbols (tipycally, yxNs, where N is an integer)
        covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: self.get_covariance_symbol(problem_moments,x,y))

        # Variances is the diagonal of covariance matrix
        variance_symbols = [covariance_matrix[i, i] for i in range(n_species)]

        # the symbols for expectations are simply the first order raw moments.
        expectation_symbols = [pm.symbol for pm in problem_moments if pm.order == 1]

        log_variance_symbols = sp.Matrix([sp.log(sp.Integer(1) + v/(e ** sp.Integer(2))) for e,v in zip(expectation_symbols, variance_symbols)])

        log_expectation_symbols = sp.Matrix([sp.log(e) - lv/sp.Integer(2) for e,lv in zip(expectation_symbols, log_variance_symbols)])

        log_variance_mat = sp.Matrix(n_species,n_species, lambda x,y: log_variance_symbols[x] if x == y else 0)

        log_covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: \
                self.get_log_covariance(log_variance_mat, log_expectation_symbols, covariance_matrix, x, y))

        # the n_vectors (e.g. [0,2,0]) of the central moments
        pm_n_vecs = [sp.Matrix(pm.n_vector) for pm in problem_moments if pm.order > 1 ]

        out_mat = sp.Matrix([n * (log_covariance_matrix * n.T) / sp.Integer(2) + n * log_expectation_symbols for n in pm_n_vecs])
        # return the exponential of all values
        out_mat = out_mat.applyfunc(lambda x: sp.exp(x))
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
