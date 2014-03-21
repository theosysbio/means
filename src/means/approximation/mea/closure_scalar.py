"""
Scalar moment closure
------

This part of the package provides the original
(and default) closure :class:`~means.approximation.mea.closure_scalar.ScalarClosure`
as well as the base class for all closers.

"""


import sympy as sp
from means.util.sympyhelpers import substitute_all


class ClosureBase(object):
    """
    A virtual class for closure methods. An implementation of `_compute_raw_moments()`
    must be provided in subclasses.
    """
    _max_order = None
    _min_order = 1

    def __init__(self,max_order, multivariate=True):
        """
        :param max_order: the maximal order of moments to be modelled.
        :type max_order: `int`
        :param multivariate: whether to consider covariances
        :return:
        """

        self._max_order = max_order
        self.__is_multivariate = multivariate
        if self._min_order > max_order:
            raise ValueError("This closure method requires `max_order` to be >= {0}".format(self._min_order))
    @property
    def is_multivariate(self):
        return self.__is_multivariate

    @property
    def max_order(self):
        return self._max_order

    def _compute_raw_moments(self, n_counter, k_counter):
        raise NotImplementedError("ParametricCloser is an abstract class.\
                                  `compute_closed_raw_moments()` is not implemented. ")

    def _compute_closed_central_moments(self, central_from_raw_exprs, n_counter, k_counter):
        r"""
        Computes parametric expressions (e.g. in terms of mean, variance, covariances) for all central moments
        up to max_order + 1 order.

        :param central_from_raw_exprs: the expression of central moments in terms of raw moments
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.core.descriptors.Moment`]
        :return: the central moments where raw moments have been replaced by parametric expressions
        :rtype: `sympy.Matrix`
        """

        closed_raw_moments = self._compute_raw_moments(n_counter, k_counter)
        assert(len(central_from_raw_exprs) == len(closed_raw_moments))
        # raw moment lef hand side symbol
        raw_symbols = [raw.symbol for raw in k_counter if raw.order > 1]

        # we want to replace raw moments symbols with closed raw moment expressions (in terms of variances/means)
        substitution_pairs = zip(raw_symbols, closed_raw_moments)
        # so we can obtain expression of central moments in terms of low order raw moments
        closed_central_moments = substitute_all(central_from_raw_exprs, substitution_pairs)
        return closed_central_moments


    def close(self, mfk, central_from_raw_exprs, n_counter, k_counter):

        """
        In MFK, replaces symbol for high order (order == max_order+1) by parametric expressions.
        That is expressions depending on lower order moments such as means, variances, covariances and so on.

        :param mfk: the right hand side equations containing symbols for high order central moments
        :param central_from_raw_exprs: expressions of central moments in terms of raw moments
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.core.descriptors.Moment`]
        :return: the modified MFK
        :rtype: `sympy.Matrix`
        """

        # we obtain expressions for central moments in terms of variances/covariances
        closed_central_moments = self._compute_closed_central_moments(central_from_raw_exprs, n_counter, k_counter)
        # set mixed central moment to zero iff univariate
        closed_central_moments = self._set_mixed_moments_to_zero(closed_central_moments, n_counter)

        # retrieve central moments from problem moment. Typically, :math: `[yx2, yx3, ...,yxN]`.

        # now we want to replace the new mfk (i.e. without highest order moment) any
        # symbol for highest order central moment by the corresponding expression (computed above)

        positive_n_counter = [n for n in n_counter if n.order > 0]
        substitutions_pairs = [(n.symbol, ccm) for n,ccm in
                               zip(positive_n_counter, closed_central_moments) if n.order > self.max_order]
        new_mfk = substitute_all(mfk, substitutions_pairs)

        return new_mfk

    def _set_mixed_moments_to_zero(self, closed_central_moments, n_counter):
        r"""
        In univariate case, set the cross-terms to 0.

        :param closed_central_moments: matrix of closed central moment
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :return:  a matrix of new closed central moments with cross-terms equal to 0
        """

        positive_n_counter = [n for n in n_counter if n.order > 1]
        if self.is_multivariate:
            return closed_central_moments
        else:
            return [0 if n.is_mixed else ccm for n,ccm in zip(positive_n_counter, closed_central_moments)]


class ScalarClosure(ClosureBase):
    """
    A class providing scalar closure to
    :class:`~means.approximation.mea.moment_expansion_approximation.MomentExpansionApproximation`.
    Expression for higher order (max_order + 1) central moments are set to a scalar.
    Typically, higher order central moments are replaced by zero.
    """
    def __init__(self,max_order,value=0):
        """
        :param max_order: the maximal order of moments to be modelled.
        :type max_order: `int`
        :param value: a scalar value for higher order moments
        """
        super(ScalarClosure, self).__init__(max_order, False)
        self.__value = value

    @property
    def value(self):
        return self.__value
    def _compute_closed_central_moments(self, central_from_raw_exprs, n_counter, k_counter):
        r"""
        Replace raw moment terms in central moment expressions by parameters (e.g. mean, variance, covariances)

        :param central_from_raw_exprs: the expression of central moments in terms of raw moments
        :param n_counter: a list of :class:`~means.core.descriptors.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.core.descriptors.Moment`]
        :param k_counter: a list of :class:`~means.core.descriptors.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.core.descriptors.Moment`]
        :return: the central moments where raw moments have been replaced by parametric expressions
        :rtype: `sympy.Matrix`
        """

        closed_central_moments = sp.Matrix([sp.Integer(self.__value)] * len(central_from_raw_exprs))
        return closed_central_moments