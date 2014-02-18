import sympy as sp
import operator
import copy
from means.util.sympyhelpers import substitute_all
from zero_closer import CloserBase



import itertools

class GammaCloser(CloserBase):
    def __init__(self,n_moments, multivariate = True):
        super(GammaCloser, self).__init__(n_moments)
        self.__is_multivariate = multivariate

    @property
    def is_multivariate(self):
        return self.__is_multivariate

    def close(self,central_moments_exprs, dmu_over_dt, central_from_raw_exprs, species, n_counter, k_counter):
        mfk = self.generate_mass_fluctuation_kinetics(central_moments_exprs, dmu_over_dt, n_counter)
        prob_lhs = self.generate_problem_left_hand_side(n_counter, k_counter)

        mfk, prob_lhs = self.parametric_closer_wrapper(mfk, central_from_raw_exprs, species, k_counter, prob_lhs)
        return mfk, prob_lhs

    def get_parameter_symbols(self, gamma_type, n_species, prob_moments ):
        #fixme set cross terms to zero for gamma type 1
        # Create symbolic species Y0 - Yn, where n = n_species
        symbolic_species = sp.Matrix([sp.Symbol('Y_{0}'.format(str(i))) for i in range(n_species + 1)])

        # Obtain beta terms in the gamma matrix
        if gamma_type == 1:
            beta_in_matrix = sp.Matrix([symbolic_species[0]] + [Y + symbolic_species[0] for Y in symbolic_species[1:]])
        elif gamma_type == 2:
            beta_in_matrix = sp.Matrix([sum(symbolic_species[0:i+1]) for i in range(n_species + 1)])
        else:
            beta_in_matrix = sp.Matrix(symbolic_species[1:])

        expectation_symbols = sp.Matrix([n.symbol for n in prob_moments if n.order == 1])
        #fixme n has no property is_mixed
        variance_symbols = sp.Matrix([n.symbol for n in prob_moments if n.order == 2 and n.is_mixed == False])

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
        elif gamma_type == 2:
                alpha_exprs_0 = sp.Matrix([1] + [alpha_bar_exprs[0] - 1])
                alpha_exprs = sp.Matrix(alpha_bar_exprs[1:]) - sp.Matrix(alpha_bar_exprs[0:len(alpha_bar_exprs)-1])
                alpha_exprs = alpha_exprs_0.col_join(alpha_exprs)

        alpha_multipliers = []
        beta_multipliers = []
        for row in counter:
            if sum(row) == 0:
                continue
            alpha_multipliers.append(reduce(operator.mul, [(a ** r).expand() for a,r in zip(beta_in_matrix[1:],row)]))
            beta_multipliers.append(reduce(operator.mul, [(b ** r).expand() for b,r in zip(beta_exprs,row)]))

        alpha_multipliers = sp.Matrix(alpha_multipliers)

        ## get alpha-expressions
        for i,a in enumerate(alpha_exprs):
            Y_to_substitute = [sp.Symbol("Y_{0}".format(i))**n for n in range(2, n_moment + 1)]
            alpha_m = [gamma_factorial(a,n) for n in range(2, n_moment +1)]

            subs_pairs = zip(Y_to_substitute, alpha_m)
            subs_pairs.append((sp.Symbol("Y_{0}".format(i)),a ))
            alpha_multipliers = alpha_multipliers.applyfunc(lambda x : substitute_all(x, subs_pairs))











    def compute_closed_central_moments(self, n_species, problem_moments):

        covariance_matrix = sp.Matrix(n_species,n_species, lambda x,y: self.get_covariance_symbol(problem_moments,x,y))


        out_mat = []

        n_counter = [n for n in problem_moments if n.order > 1]

        for n in n_counter:

            list_for_partition = []

            for i,e in enumerate(n.n_vector):
                list_for_partition.extend([i for n_times in range(e)])


            if n.order % 2 != 0:
                each_row = [sp.Integer(0)]

            elif n.order == 2:

                each_row = [covariance_matrix[list_for_partition[0],list_for_partition[1]]]
            else:
               #fixme
                # retrieve the items based on the indices pairs and add the partitions
                idx = self.partition(2,[[]],0,list_for_partition)
                #print "idx"
                #print idx

                each_row = []
                #print "idx_pairs"
                #for idx_pairs in self.generate_partitions(list_for_partition):
                for idx_pairs in idx:

                    l = [covariance_matrix[i[0],i[1]] for i in idx_pairs]
                    row_elements = reduce(operator.mul,l)
                    each_row.append(row_elements)

            out_mat.append(sum(each_row))

        return out_mat



    def parametric_closer_wrapper(self, mfk, central_from_raw_exprs, species, k_counter, prob_moments):

        n_moments = self.n_moments
        n_species = len(species)

        # we obtain expressions for central moments in terms of variances/covariances
        closed_central_moments = self.compute_closed_central_moments(n_species, prob_moments)
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
        idxs = [i for i in range(len(l))]



        # make all combinations
        # A natural property of these combinaisons, is that the last element
        # is a complementary set to the fist one, the second to the one before last,
        # and so on.
        combin =  [i for i in itertools.combinations(idxs , part_size)]

        # this loop will return the first, the second with the one before last ans so on
        # eg (0,1,2,3,4,5) ->  (0,5),(1,4),(2,3)
        for i in combin:
            yield (i,combin.pop())

    def partition(self, k, accum, index, list_for_par):
        '''
        :param k: the number of elements in each partition
        :param accum: should be [[]] as each partition pair consists of lists within a list
        :param index: the index of item in list to start partition.should start from 0
        :param list_for_par: the list for partition
        :return: a list of non-repetitive partition pairs, each partition pair contains 2 indeces for variance
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
            result.extend(self.partition(k - 1, clone_new, index + 1, list_for_par))

            for elem_i in range(len(accum[set_i])):
                clone_new = copy.deepcopy(accum)
                clone_new[set_i][elem_i].append(element)
                result.extend(self.partition(k, clone_new, index + 1, list_for_par))

        return [row for row in result if len(row[0]) == 2]