import sympy as sp

class CloserBase(object):

    _n_moments = None
    def __init__(self,n_moments):
        self._n_moments = n_moments
    @property
    def n_moments(self):
        return self._n_moments

    def close(self,central_moments_exprs, dmu_over_dt, central_from_raw_exprs, species, n_counter, k_counter):
        raise NotImplementedError("CloserBase is an abstract class. `close()` is not implemented. ")

    def generate_problem_left_hand_side(self, n_counter, k_counter):
        # concatenate the symbols for first order raw moments (means)
        prob_moments_over_dt = [k for k in k_counter if k.order == 1]
        # and the higher order central moments (variances, covariances,...)
        prob_moments_over_dt += [n for n in n_counter if n.order > 1]
        return prob_moments_over_dt

    def generate_mass_fluctuation_kinetics(self, central_moments, dmu_over_dt, n_counter):
        """
        :param central_moments:
        :param n_counter:
        :param dmu_over_dt:
        :return: the right hand side of the final ODEs
        """
        # symbols for central moments
        central_moments_symbols = sp.Matrix([n.symbol for n in n_counter])
        # try to simplify an expression. returns the original expression if fail
        # # todo remove this when we do not need it anymore
        # def try_to_simplify(expr):
        #     try:
        #         return sp.simplify(expr)
        #     except:
        #         pass
        #     return expr
        #

        # rhs for the first order raw moment
        mfk = [e for e in dmu_over_dt * central_moments_symbols]
        # rhs for the higher order raw moments
        mfk += [(sp.Matrix(cm).T * central_moments_symbols)[0] for cm in central_moments.tolist()]
        return sp.Matrix(mfk)

class ZeroCloser(CloserBase):
    def close(self,central_moments_exprs, dmu_over_dt, central_from_raw_exprs, species, n_counter, k_counter):
        mfk = self.generate_mass_fluctuation_kinetics(central_moments_exprs, dmu_over_dt, n_counter)
        prob_lhs = self.generate_problem_left_hand_side(n_counter, k_counter)
        return mfk, prob_lhs
