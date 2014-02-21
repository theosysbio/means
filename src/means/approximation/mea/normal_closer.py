import sympy as sp
import operator
from means.util.sympyhelpers import product
from zero_closer import ParametricCloser



import itertools

class NormalCloser(ParametricCloser):

    def get_covariance_symbol(self, q_counter, sp1_idx, sp2_idx):
        '''
        Compute second order moments i.e. variances and covariances
        Covariances equal to 0 in univariate case
        :param q_counter: moment matrix
        :param sp1_idx: index of one species
        :param sp2_idx: index of another species
        :return: second order moments matrix of size n_species by n_species
        '''

        # The diagonal positions in the matrix are the variances
        if sp1_idx == sp2_idx:
            return [q.symbol for q in q_counter if q.n_vector[sp1_idx] == 2 and q.order == 2][0]

        # In multivariate cases, return covariances
        elif self.is_multivariate:
            return [q.symbol for q in q_counter if q.n_vector[sp1_idx] == 1 and q.n_vector[sp2_idx] == 1 and q.order == 2][0]

        # In univariate cases, covariances are 0s
        else:
            return sp.Integer(0)


    def compute_one_closed_central_moment(self, moment, covariance_matrix):
        '''
        Compute each row of closed central moment based on Isserlis' Theorem of calculating higher order moments
        of multivariate normal distribution in terms of covariance matrix

        :param moment: moment matrix
        :param covariance_matrix: matrix containing variances and covariances
        :return:  each row of closed central moment
        '''

        # If moment order is odd, higher order moments equals 0
        if moment.order % 2 != 0:
            return sp.Integer(0)

        # index of species
        idx = [i for i in range(len(moment.n_vector))]

        # repeat the index of a species as many time as its value in counter
        list_for_partition = reduce(operator.add, map(lambda i, c: [i] * c, idx, moment.n_vector))
        print list_for_partition
        print "=>"
        # If moment order is even, :math: '\mathbb{E} [x_1x_2 \ldots  x_2_n] = \sum \prod\mathbb{E} [x_ix_j] '
        # e.g.:math: '\mathbb{E} [x_1x_2x_3x_4] = \mathbb{E} [x_1x_2] +\mathbb{E} [x_1x_3] +\mathbb{E} [x_1x_4]
        # +\mathbb{E} [x_2x_3]+\mathbb{E} [x_2x_4]+\mathbb{E} [x_3x_4]'
        # For second order moment, there is only one way of partitioning. Hence, no need to generate partitions
        if moment.order == 2:
            return covariance_matrix[list_for_partition[0], list_for_partition[1]]

        # For even moment order other than 2, generate a list of partitions of the indices of covariances
        else:
            each_row = []
            for idx_pair in self.generate_partitions(list_for_partition):
                # Retrieve the pairs of covariances using the pairs of partitioned indices
                l = [covariance_matrix[i, j] for i,j in idx_pair]
                # Calculate the product of each pair of covariances
                each_row.append(product(l))

            # The corresponding closed central moment of that moment order is the sum of the products
            return sum(each_row)


    def compute_closed_central_moments(self, central_from_raw_exprs, k_counter, problem_moments):
        n_species = len([None for pm in problem_moments if pm.order == 1])
        covariance_matrix = sp.Matrix(n_species, n_species, lambda x,y: self.get_covariance_symbol(problem_moments,x,y))
        n_counter = [n for n in problem_moments if n.order > 1]
        out_mat = [self.compute_one_closed_central_moment(n, covariance_matrix) for n in n_counter]
        return sp.Matrix(out_mat)


    def generate_partitions(self, l):

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