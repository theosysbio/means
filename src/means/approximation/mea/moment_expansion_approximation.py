from assimulo import problem
import itertools
import sympy as sp
from means.approximation import ode_problem
from means.approximation.approximation_baseclass import ApproximationBaseClass
from means.approximation.ode_problem import Moment
from TaylorExpansion import generate_dmu_over_dt
from centralmoments import eq_centralmoments
from raw_to_central import raw_to_central
from means.util.sympyhelpers import substitute_all
from gamma_closer import GammaCloser
from log_normal_closer import LogNormalCloser
from normal_closer import NormalCloser
from zero_closer import  ZeroCloser

class MomentExpansionApproximation(ApproximationBaseClass):
    """
    Performs moment expansion approximation (Ale et al. 2013) up to a given order of moment.
    """
    def __init__(self, model, n_moments, closer='zero', *closer_args, **closer_kwargs):
        super(MomentExpansionApproximation, self).__init__(model)
        self.__n_moments = int(n_moments)

        # a dictionary of "option -> closer" this allow a generic handling for closer without to have to add
        # if-else and exceptions when implementing new closers. One only needs to add the new closer class to the dict
        supported_closers = {"log-normal": LogNormalCloser,
                             "zero": ZeroCloser,
                             "normal": NormalCloser,
                             "gamma": GammaCloser}

        # exception it the closer name is not in the dict
        if not closer in supported_closers:
            error_str = "The closer type '{0}' is not supported.\n Supported values for closer:\n{1}"
            raise KeyError(error_str.format(closer,supported_closers))
        # otherwise, we initialise the closer for this approximator
        else:
            # our closer is an instance of the class queried in the dictionary
            CloserClass = supported_closers[closer]
            self.__closer = CloserClass(n_moments, *closer_args, **closer_kwargs)

    @property
    def closer(self):
        return self.__closer

    def run(self):
        """
        Overrides the default run() method.
        Performs the complete analysis
        :return: an ODEProblem which can be further used in inference and simulation
        """
        n_moments = self.__n_moments
        stoichiometry_matrix = self.model.stoichiometry_matrix
        propensities = self.model.propensities
        species = self.model.species
        # compute n_counter and k_counter; the "n" and "k" vectors in equations, respectively.
        n_counter, k_counter = self.generate_n_and_k_counters(n_moments, species)
        # dmu_over_dt has row per species and one col per element of n_counter (eq. 6)
        dmu_over_dt = generate_dmu_over_dt(species, propensities, n_counter, stoichiometry_matrix)
        #  Calculate expressions to use in central moments equations (eq. 9)
        central_moments_exprs = eq_centralmoments(n_counter, k_counter, dmu_over_dt, species, propensities, stoichiometry_matrix)
        # Expresses central moments in terms of raw moments (and central moments) (eq. 8)
        central_from_raw_exprs = raw_to_central(n_counter, species, k_counter)
        # Substitute raw moment, in central_moments, with expressions depending only on central moments
        central_moments_exprs = self.substitute_raw_with_central(central_moments_exprs, central_from_raw_exprs, n_counter, k_counter)
        # Get final right hand side expressions for each moment in a vector
        mfk, prob_lhs = self.closer.close(central_moments_exprs, dmu_over_dt, central_from_raw_exprs, species, n_counter, k_counter)

        out_problem = ode_problem.ODEProblem("MEA", prob_lhs, mfk, sp.Matrix(self.model.constants))
        return out_problem

    def substitute_raw_with_central(self, central_moments_exprs, central_from_raw_exprs, n_counter, k_counter):
        """
        Takes the expressions for central moments, and substitute the symbols representing raw moments,
        by equivalent expressions in terms of central moment
        :param central_moments_exprs: a matrix of expressions for central moments.
        :param central_from_raw_exprs: central moment expressed in terms of raw moments
        :param n_counter: the counter for central moments
        :param k_counter: the counter for raw moments
        :return: expression of central moments without raw moment
        """
        positiv_raw_moms_symbs = [raw.symbol for raw in k_counter if raw.order > 1]
        # The symbols for the corresponding central moment
        central_symbols= [central.symbol for central in n_counter if central.order > 1]
        # Now we state (central_symbols == central_from_raw_exprs)
        eq_to_solve = [sp.Eq(cfr, cs) for (cs, cfr) in zip(central_symbols, central_from_raw_exprs)]
        # And we solve this for the symbol of the corresponding raw moment. This gives an expression
        # of the symbol for raw moment in terms of central moments and lower order raw moment
        solved_xs = sp.Matrix([sp.solve(rhs, raw) for (rhs, raw) in zip(eq_to_solve, positiv_raw_moms_symbs)])

        # now we want to express raw moments only in terms od central moments and means
        # for instance if we have: :math:`x_1 = 1, x_2 = 2 +x_1, x_3 = x_2*x_1`, we should give:
        # :math: `x_1 = 1, x_2 = 2+1, x_3 = 1*(2+1)`
        # To achieve this, we recursively apply substitution as many times as the highest order (minus one)
        max_order = max([p.order for p in k_counter])

        for i in range(max_order - 1):
            substitution_pairs = zip(positiv_raw_moms_symbs, solved_xs)
            solved_xs = substitute_all(solved_xs, substitution_pairs)

        # we finally build substitution pairs to replace all raw moments
        substitution_pairs = zip(positiv_raw_moms_symbs, solved_xs)

        # apply this substituton to all elements of the central moment expressions matrix
        out_exprs = substitute_all(central_moments_exprs, substitution_pairs)

        #todo eventually, remove simplify (slow)
        #out_exprs = out_exprs.applyfunc(sp.simplify)
        return out_exprs

    def generate_n_and_k_counters(self, n_moments, species, central_symbols_prefix="yx", raw_symbols_prefix="x_"):
        """
        Makes a counter for central moments (n_counter) and a counter for raw moment (k_counter)
        Each is a list of "Moment" objects. Therefore, they are represented by both a vector of integer
        and a symbol.

        :param n_moments: the maximal order of moment to be computer
        :param species: the name of the species
        :return: a pair of lists of Moments
        """

        # first order moments are always 1
        k_counter = [Moment([0] * len(species), sp.Integer(1))]
        n_counter = [Moment([0] * len(species), sp.Integer(1))]

        # build descriptors for first order raw moments aka expectations (e.g. [1, 0, 0], [0, 1, 0] and [0, 0, 1])
        descriptors = []
        for i in range(len(species)):
            row = [0]*len(species)
            row[i] = 1
            descriptors.append(row)

        # We use species name as symbols for first order raw moment
        k_counter += [Moment(d, s) for d,s in zip(descriptors, species)]

        # Higher order raw moment descriptors
        k_counter_descriptors = [i for i in itertools.product(range(n_moments + 1), repeat=len(species))
                                 if 1 < sum(i) <= n_moments]

        #this mimics matlab sorting
        k_counter_descriptors = sorted(k_counter_descriptors,lambda x,y: sum(x) - sum(y))
        #k_counter_descriptors = [[r for r in reversed(k)] for k in k_counter_descriptors]
        k_counter_symbols = [sp.Symbol(raw_symbols_prefix + "_".join([str(s) for s in count]),real=True)
                             for count in k_counter_descriptors]
        k_counter += [Moment(d, s) for d,s in zip(k_counter_descriptors, k_counter_symbols)]

        #  central moments
        n_counter_descriptors = [m for m in k_counter_descriptors if sum(m) > 1]
        # arbitrary symbols
        n_counter_symbols = [sp.Symbol(central_symbols_prefix + str(i+1),real=True) for i in range(len(n_counter_descriptors))]
        n_counter += [Moment(c, s) for c,s in zip(n_counter_descriptors, n_counter_symbols)]

        return n_counter, k_counter

