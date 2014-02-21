import sympy as sp
import operator
import copy
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

        # If moment order is even, :math: '\mathbb{E} [x_1x_2 \ldots  x_2_n] = \sum \prod\mathbb{E} [x_ix_j] '
        # e.g.:math: '\mathbb{E} [x_1x_2x_3x_4] = \mathbb{E} [x_1x_2] +\mathbb{E} [x_1x_3] +\mathbb{E} [x_1x_4]
        # +\mathbb{E} [x_2x_3]+\mathbb{E} [x_2x_4]+\mathbb{E} [x_3x_4]'
        # For second order moment, there is only one way of partitioning. Hence, no need to generate partitions
        if moment.order == 2:
            return covariance_matrix[list_for_partition[0], list_for_partition[1]]

        # For even moment order other than 2, generate a list of partitions of the indices of covariances
        else:
            each_row = []
            for idx_pair in self.generate_partitions(len(list_for_partition)/2,list_for_partition):
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

    def generate_partitions(self, k, list_for_par, accum=[[]], index=0):
        '''
        :param list_for_par: the list for partition
        :param accum: should be [[]] as each partition pair consists of lists within a list
        :param index: the index of item in list to start partition.should start from 0
        :return: a list of non-repetitive partition pairs, each partition pair contains 2 indices for variance
        '''
        if index == len(list_for_par):
            if (k == 0):
                return accum
            else:
                return []

        element = list_for_par[index]
        result = []

        for set_i in range(len(accum)):
            clone_new = copy.deepcopy(accum)
            clone_new[set_i].append([element])
            result.extend(self.generate_partitions(k - 1, list_for_par, clone_new, index + 1))

            for elem_i in range(len(accum[set_i])):
                clone_new = copy.deepcopy(accum)
                clone_new[set_i][elem_i].append(element)
                result.extend(self.generate_partitions(k, list_for_par,clone_new, index + 1))

        return [row for row in result if all([(len(r) == 2) for r in row])]






