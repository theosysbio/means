import sympy as sp
from zero_closer import CloserBase

class LogNormalCloser(CloserBase):

    def compute_raw_moments(self, n_counter, k_counter):

        # The symbols for expectations are simply the first order raw moments.
        expectation_symbols = [pm.symbol for pm in k_counter if pm.order == 1]

        n_species = len(expectation_symbols)

        # The covariance expressed in terms of central moment symbols (typically, yxNs, where N is an integer)
        covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: self.get_covariance_symbol(n_counter,x,y))

        # Variances is the diagonal of covariance matrix
        variance_symbols = [covariance_matrix[i, i] for i in range(n_species)]

        # :math: '\logVar(x_i) = 1 + \frac { Var(x_i)}{ \mathbb{E} (x_i)^2}'
        log_variance_symbols = sp.Matrix([sp.log(sp.Integer(1) + v/(e ** sp.Integer(2))) for e,v in zip(expectation_symbols, variance_symbols)])

        # :math: '\log\mathbb{E} (x_i) = \log(\mathbb{E} (x_i) )+ \frac {\log (Var(x_i))}{2}'
        log_expectation_symbols = sp.Matrix([sp.log(e) - lv/sp.Integer(2) for e,lv in zip(expectation_symbols, log_variance_symbols)])

        # Assign log variance symbols on the diagonal of size n_species by n_species
        log_variance_mat = sp.Matrix(n_species,n_species, lambda x,y: log_variance_symbols[x] if x == y else 0)

        # Assign log covariances and log variances in the matrix log_covariance matrix based on matrix indices
        log_covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: \
                self.get_log_covariance(log_variance_mat, log_expectation_symbols, covariance_matrix, x, y))

        # The n_vectors (e.g. [0,2,0]) of the central moments
        pm_n_vecs = [sp.Matrix(pm.n_vector) for pm in n_counter if pm.order > 1 ]

        #todo find out the equation
        out_mat = sp.Matrix([n * (log_covariance_matrix * n.T) / sp.Integer(2) + n * log_expectation_symbols for n in pm_n_vecs])

        # return the exponential of all values
        out_mat = out_mat.applyfunc(lambda x: sp.exp(x))
        return out_mat

    def get_covariance_symbol(self, q_counter, sp1_idx, sp2_idx):
        r"""
        Compute second order moments i.e. variances and covariances
        Covariances equal to 0 in univariate case

        :param q_counter: moment matrix
        :param sp1_idx: index of one species
        :param sp2_idx: index of another species
        :return: second order moments matrix of size n_species by n_species
        """
        # The diagonal positions in the matrix are the variances
        if sp1_idx == sp2_idx:
            return [q.symbol for q in q_counter if q.n_vector[sp1_idx] == 2 and q.order == 2][0]
        # Covariances are found if the moment order is 2 and the moment vector contains double 1
        return [q.symbol for q in q_counter if q.n_vector[sp1_idx] == 1 and q.n_vector[sp2_idx] == 1 and q.order == 2][0]

    def get_log_covariance(self, log_variance_mat, log_expectation_symbols, covariance_matrix, x, y):
        r"""
        Compute log covariances according to:\\

        :math:`\log{(Cov(x_i,x_j))} = \frac { 1 + Cov(x_i,x_j)}{\exp[\log \mathbb{E}(x_i) + \log \mathbb{E}(x_j)+\frac{1}{2} (\log Var(x_i) + \log Var(x_j)]}`

        :param log_variance_mat: a column matrix of log variance
        :param log_expectation_symbols: a column matrix of log expectations
        :param covariance_matrix: a matrix of covariances
        :param x: x-coordinate in matrix of log variances and log covariances
        :param y: y-coordinate in matrix of log variances and log covariances
        :return: a matrix of log covariances and log variances
        """
        # The diagonal of the return matrix includes all the log variances
        if x == y:
            return log_variance_mat[x,x]
        # log covariances are calculated if not on the diagonal of the return matrix
        elif self.is_multivariate:
            denom = sp.exp(log_expectation_symbols[x] +
                           log_expectation_symbols[y] +
                           (log_variance_mat[x,x] + log_variance_mat[y, y])/ sp.Integer(2))
            return sp.log(sp.Integer(1) + covariance_matrix[x, y] / denom)
        # univariate case: log covariances are 0s.
        else:
            return sp.Integer(0)



