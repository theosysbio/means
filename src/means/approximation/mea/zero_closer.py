import sympy as sp
from means.util.sympyhelpers import substitute_all

class CloserBase(object):

    _max_order = None
    def __init__(self,max_order,multivariate=True):
        self._max_order = max_order
        self.__is_multivariate = multivariate


    @property
    def is_multivariate(self):
        return self.__is_multivariate

    @property
    def max_order(self):
        return self._max_order

    def compute_raw_moments(self, n_counter, k_counter):
        raise NotImplementedError("ParametricCloser is an abstract class.\
                                  `compute_closed_raw_moments()` is not implemented. ")

    def compute_closed_central_moments(self, central_from_raw_exprs, n_counter, k_counter):
        """
        Replace raw moment terms in central moment expressions by parameters (e.g. mean, variance, covariances)

        :param closed_raw_moments: the expression of all raw moments (expect 0th) in terms of
        parameters such as variance/covariance (i.e. central moments) and first order raw moment (i.e. means)
        :param central_from_raw_exprs: the expression of central moments in terms of raw moments
        :param k_counter: a list of `Moment` object corresponding to raw moment symbols an descriptors
        :return: the central moments where raw moments have been replaced by parametric expressions
        :rtype: sympy.Matrix
        """

        closed_raw_moments = self.compute_raw_moments(n_counter, k_counter)
        # raw moment lef hand side symbol
        raw_symbols = [raw.symbol for raw in k_counter if raw.order > 1]
        # we want to replace raw moments symbols with closed raw moment expressions (in terms of variances/means)
        substitution_pairs = zip(raw_symbols, closed_raw_moments)
        # so we can obtain expression of central moments in terms of low order raw moments
        closed_central_moments = substitute_all(central_from_raw_exprs, substitution_pairs)
        return closed_central_moments


    def close(self, mfk, central_from_raw_exprs, n_counter, k_counter):

        # we obtain expressions for central moments in terms of variances/covariances
        closed_central_moments = self.compute_closed_central_moments(central_from_raw_exprs, n_counter, k_counter)
        # set mixed central moment to zero iff univariate
        closed_central_moments = self.set_mixed_moments_to_zero(closed_central_moments, n_counter)

        # retrieve central moments from problem moment. Typically, :math: `[yx2, yx3, ...,yxN]`.

        # now we want to replace the new mfk (i.e. without highest order moment) any
        # symbol for highest order central moment by the corresponding expression (computed above)

        positive_n_counter = [n for n in n_counter if n.order > 0]
        substitutions_pairs = [(n.symbol, ccm) for n,ccm in
                               zip(positive_n_counter, closed_central_moments) if n.order > self.max_order]
        new_mfk = substitute_all(mfk, substitutions_pairs)

        return new_mfk

    def set_mixed_moments_to_zero(self, closed_central_moments, n_counter):
        '''
        In univariate case, set the cross-terms to 0.
        :param closed_central_moments: matrix of closed central moment
        :param prob_moments: moment matrix
        :return:  a matrix of new closed central moments with cross-terms equal to 0
        '''
        positive_n_counter = [n for n in n_counter if n.order > 1]
        if self.is_multivariate:
            return closed_central_moments
        else:
            return [0 if n.is_mixed else ccm for n,ccm in zip(positive_n_counter, closed_central_moments)]


class ZeroCloser(CloserBase):
    def compute_closed_central_moments(self, central_from_raw_exprs, n_counter, k_counter):
        """
        Replace raw moment terms in central moment expressions by parameters (e.g. mean, variance, covariances)

        :param closed_raw_moments: the expression of all raw moments (expect 0th) in terms of
        parameters such as variance/covariance (i.e. central moments) and first order raw moment (i.e. means)
        :param central_from_raw_exprs: the expression of central moments in terms of raw moments
        :param k_counter: a list of `Moment` object corresponding to raw moment symbols an descriptors
        :return: the central moments where raw moments have been replaced by parametric expressions
        :rtype: sympy.Matrix
        """

        closed_central_moments = sp.Matrix([sp.Integer(0)] * len(central_from_raw_exprs))
        return closed_central_moments