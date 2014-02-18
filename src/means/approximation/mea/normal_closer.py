import sympy as sp
import operator
import copy
from means.util.sympyhelpers import substitute_all
from means.util.sympyhelpers import product
from zero_closer import CloserBase



import itertools

class NormalCloser(CloserBase):
    def __init__(self,n_moments, multivariate = True):
        super(NormalCloser, self).__init__(n_moments)
        self.__is_multivariate = multivariate

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
        elif self.is_multivariate:
            return [q.symbol for q in q_counter if q.n_vector[sp1_idx] == 1 and q.n_vector[sp2_idx] == 1 and q.order == 2][0]
        else:
            return sp.Integer(0)


    def compute_one_closed_central_moment(self, moment, covariance_matrix):

        if moment.order % 2 != 0:
            return sp.Integer(0)

        # index of species
        idx = [i for i in range(len(moment.n_vector))]
        # repeat the index of a species as many time as its value in counter
        list_for_partition = reduce(operator.add, map(lambda i, c: [i] * c, idx, moment.n_vector))

        if moment.order == 2:
            return covariance_matrix[list_for_partition[0], list_for_partition[1]]

        else:
            each_row = []
            for idx_pair in self.generate_partitions(list_for_partition):
                l = [covariance_matrix[i, j] for i,j in idx_pair]
                each_row.append(product(l))

            return sum(each_row)


    def compute_closed_central_moments(self, n_species, problem_moments):
        covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: self.get_covariance_symbol(problem_moments,x,y))
        n_counter = [n for n in problem_moments if n.order > 1]
        out_mat = [self.compute_one_closed_central_moment(n, covariance_matrix) for n in n_counter]
        return sp.Matrix(out_mat)


    def set_mixed_moments_to_zero(self, closed_central_moments,prob_moments):
        n_counter = [n for n in prob_moments if n.order > 1]
        if self.is_multivariate:
            return closed_central_moments
        else:
            return [0 if n.is_mixed else ccm for n,ccm in zip(n_counter, closed_central_moments)]


    def parametric_closer_wrapper(self, mfk, central_from_raw_exprs, species, k_counter, prob_moments):

        n_moments = self.n_moments
        n_species = len(species)

        # we obtain expressions for central moments in terms of variances/covariances
        closed_central_moments = self.compute_closed_central_moments(n_species, prob_moments)
        # set mixed central moment to zero iff univariate
        closed_central_moments = self.set_mixed_moments_to_zero(closed_central_moments,prob_moments)
        # we remove ODEs of highest order in mfk
        new_mfk = sp.Matrix([mfk for mfk, pm in zip(mfk, prob_moments) if pm.order < n_moments])

        # retrieve central moments from problem moment. Typically, :math: `[yx2, yx3, ...,yxN]`.
        n_counter = [n for n in prob_moments if n.order > 1]
        # now we want to replace the new mfk (i.e. without highest order moment) any
        # symbol for highest order central moment by the corresponding expression (computed above)
        substitutions_pairs = [(n.symbol, ccm) for n,ccm in zip(n_counter, closed_central_moments) if n.order == n_moments]
        new_mfk = substitute_all(new_mfk, substitutions_pairs)
        # we also update problem moments (aka lhs) to match remaining rhs (aka mkf)
        new_prob_moments = [pm for pm in prob_moments if pm.order < n_moments]


        return new_mfk,new_prob_moments

    def generate_partitions(self,l):

        if len(l) % 2 != 0:
            raise ValueError("the length of the list to partition must be even")
        #define partition size
        part_size = int(len(l)/2)
        #idxs = [i for i in range(len(l))]

        # make all combinations
        # A natural property of these combinations, is that the last element
        # is a complementary set to the fist one, the second to the one before last,
        # and so on.
        combin = [i for i in itertools.combinations(l, part_size)]

        # this loop will return the first, the second with the one before last ans so on
        # eg (0,1,2,3,4,5) ->  (0,5),(1,4),(2,3)
        for i in combin:
            yield (i,combin.pop())