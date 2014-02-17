import sympy as sp
import operator
import copy
from means.util.sympyhelpers import substitute_all
from zero_closer import CloserBase
class NormalCloser(CloserBase):
    def __init__(self,n_moments, multivariate = True):
        super(NormalCloser, self).__init__(n_moments)
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

    def partition(k,accum,index,list_for_par):
            '''
            :param k: the number of elements in each partition
            :param accum: should be [[]] as each partition pair consists of lists within a list
            :param index: the index of item in list to start partition.should start from 0
            :param list_for_par: the list for partition
            :return: a list of non-repetitive partition pairs, each partition pair contains 2 indeces for variance
            '''
            if index == len(list_for_par):
                if(k==0):
                    return accum
                else:
                    return []

            element = list_for_par[index]
            result = []

            for set_i in range(len(accum)):
                clone_new = copy.deepcopy(accum)
                clone_new[set_i].append([element])
                result.extend(partition(k-1,clone_new,index+1,list_for_par) )

                for elem_i in range(len(accum[set_i])):
                    clone_new = copy.deepcopy(accum)
                    clone_new[set_i][elem_i].append(element)
                    result.extend( partition(k,clone_new,index+1,list_for_par) )

            return [row for row in result if len(row[0]) ==2]

    def compute_raw_moments(self, n_counter, n_species, problem_moments):
        covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: self.get_covariance_symbol(problem_moments,x,y))
        variance_symbols = [covariance_matrix[i, i] for i in range(n_species)]

        print variance_symbols

        out_mat = []

        for n in n_counter:
            n.order = sum(n)
            if n.order % 2 != 0:
                out_mat.append([0])

            else:
                list_for_partition = []

                for i,e in enumerate(n):
                    list_for_partition.extend([i for n_times in range(e)])

                # retrieve the items based on the indeces pairs and add the partitions

                idx = partition(2,[[]],0,list_for_partition)
                if n.order == 2:
                    each_row = [variance_symbols[list_for_partition[0]][list_for_partition[1]]]
                else:
                    each_row = []
                    for idx_pairs in idx:
                        l = [variance_symbols[i[0]][i[1]] for i in idx_pairs]
                        row_elements = reduce(operator.mul,l)
                        each_row.append(row_elements)
                out_mat.append(sum(each_row))

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
        closed_raw_moments = self.compute_raw_moments(n_counter, n_species, prob_moments)

        # we obtain expressions for central moments in terms of closed raw moments
        closed_central_moments = self.compute_closed_central_moments(closed_raw_moments, central_from_raw_exprs, k_counter)

        # we remove ODEs of highest order in mfk
        new_mkf = sp.Matrix([mfk for mfk, pm in zip(mfk, prob_moments) if pm.order < n_moments])

        # retrieve central moments from problem moment. Typically, :math: `[yx2, yx3, ...,yxN]`.
        n_counter = [n for n in prob_moments if n.order > 1]

        # now we want to replace the new mfk (i.e. without highest order moment) any
        # symbol for highest order central moment by the corresponding expression (computed above)
        substitutions_pairs = [(n.symbol, ccm) for n,ccm in zip(n_counter, closed_central_moments) if n.order == n_moments]
        new_mkf = substitute_all(new_mkf, substitutions_pairs)

        # we also update problem moments (aka lhs) to match remaining rhs (aka mkf)
        new_prob_moments = [pm for pm in prob_moments if pm.order < n_moments]

        return new_mkf,new_prob_moments
    print new_mkf