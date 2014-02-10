import itertools
import sympy as sp
from means.approximation import ode_problem
from means.approximation.approximation_baseclass import ApproximationBaseClass
from means.approximation.ode_problem import Moment
from TaylorExpansion import generate_dmu_over_dt
from centralmoments import eq_centralmoments
from raw_to_central import raw_to_central
from log_normal_closer import log_normal_closer_wrapper

class MomentExpansionApproximation(ApproximationBaseClass):
    """
    Performs moment expansion approximation (Ale et al. 2013) up to a given order of moment.
    """
    def __init__(self, model, n_moments):
        super(MomentExpansionApproximation, self).__init__(model)
        self.__n_moments = int(n_moments)

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


        log_normal_closer_wrapper(central_from_raw_exprs, n_counter, k_counter, n_moments, species)


        # Substitute raw moment, in central_moments, with expressions depending only on central moments
        central_moments_exprs = self.substitute_raw_with_central(central_moments_exprs, central_from_raw_exprs, n_counter, k_counter)
        # Get final right hand side expressions for each moment in a vector



        mass_fluctuation_kinetics = self.generate_mass_fluctuation_kinetics(central_moments_exprs, n_counter, dmu_over_dt)
        # concatenate the first order raw moments (means)


        prob_moments = [k for k in k_counter if k.order == 1]
        # and the higher order central moments (variances, covariances,...)
        prob_moments += [n for n in n_counter if n.order > 1]
        # return a problem object


        out_problem = ode_problem.ODEProblem("MEA", prob_moments, mass_fluctuation_kinetics, sp.Matrix(self.model.constants))
        return out_problem

    def substitute_raw_with_central(self, central_moments_exprs, central_from_raw_exprs, n_counter, k_counter):
        """
        Takes the expressions for central moments, and substitute the symbols representing raw moments,
        by equivalent expressions in terms of central moment
        :param central_moments_exprs: a matrix of expressions for central moments.
        :param central_from_raw_exprs: central moment expressed in terms of raw moments
        :param n_counter:
        :param k_counter:
        :return: expression of central moments without raw moment
        """
        #fixme:
        # Here we assume the n and k counters to be in same order.
        # It would be better to ensure the moments n_vectors match
        # The symbols for raw moment symbols
        raw_lhs = [raw.symbol for raw in k_counter if raw.order > 1]
        # The symbols for the corresponding central moment
        central_symbols= [central.symbol for central in n_counter if central.order > 1]
        # Now we state (central_symbols - central_from_raw_exprs) == 0
        eq_to_solve = [cfr - cs for (cs, cfr) in zip(central_symbols, central_from_raw_exprs)]
        # And we solve this for the symbol of the corresponding raw moment. This gives an expression
        #  of the symbol for raw moment in terms of central moments and lower order raw moment
        solved_xs = [sp.solve(rhs, rlhs) for (rhs, rlhs) in zip(eq_to_solve, raw_lhs)]

        #sympy 0.7.4 compatibility
        try:
            out_exprs = central_moments_exprs.clone()
        except:
            out_exprs = central_moments_exprs.copy()

        # "reversed" since we start the substitutions by higher order moments and propagate to the lower order moments
        for rlhs, sx in reversed(zip(raw_lhs, solved_xs)):
            out_exprs = out_exprs.applyfunc(lambda x : sp.Subs(x, rlhs, sx).doit())
        #todo eventually, remove simplify (slow)
        out_exprs = out_exprs.applyfunc(sp.simplify)
        return out_exprs

    def generate_mass_fluctuation_kinetics(self, central_moments, n_counter, dmu_over_dt):
        """
        :param central_moments:
        :param n_counter:
        :param dmu_over_dt:
        :return: MFK ...
        """
        # symbols for central moments
        central_moments_symbols = sp.Matrix([n.symbol for n in n_counter])
        # try to simplify an expression. returns the original expression if fail
        # todo remove this when we do not need it anymore
        def try_to_simplify(expr):
            try:
                return sp.simplify(expr)
            except:
                pass
            return expr


        # rhs for the first order raw moment
        MFK = [try_to_simplify(e) for e in dmu_over_dt * central_moments_symbols]
        # rhs for the higher order raw moments
        MFK += [try_to_simplify((sp.Matrix(cm).T * central_moments_symbols)[0]) for cm in central_moments.tolist()]
        return sp.Matrix(MFK)

    def generate_n_and_k_counters(self, n_moments, species):
        """
        Makes a counter for central moments (n_counter) and a counter for raw moment (k_counter)
        Each is a list of "Moment" objects. Therefore, they are represented by both a vector of integer
        and a symbol.

        :param n_moments: the maximal order of moment to be computer
        :param species: the name of the species
        :return: a pair of lists of Moments
        """

        # first order moments are 1
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
        k_counter_descriptors = [i for i in itertools.product(range(n_moments + 1), repeat=len(species)) if sum(i) <= n_moments and sum(i) > 1]
        k_counter_symbols = [sp.S("x_" + "_".join([str(s) for s in count])) for count in k_counter_descriptors]
        k_counter += [Moment(d, s) for d,s in zip(k_counter_descriptors, k_counter_symbols)]

        #  central moments
        n_counter_descriptors = [m for m in k_counter_descriptors if sum(m) > 1]
        n_counter_symbols = [sp.S('yx{0}'.format(i+1)) for i in range(len(n_counter_descriptors))]
        n_counter += [Moment(c, s) for c,s in zip(n_counter_descriptors, n_counter_symbols)]

        return n_counter, k_counter

