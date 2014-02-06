import itertools
import sympy as sp
from approximation_baseclass import ApproximationBaseClass
import ode_problem
from ode_problem import Moment

from TaylorExpansion import taylor_expansion
from centralmoments import eq_centralmoments
from raw_to_central import raw_to_central

class MomentExpansionApproximation(ApproximationBaseClass):

    """
    Performs moment expansion approximation described in Ale et al. 2013) up to a given order of moment.
    """

    def __init__(self, model, n_moments):
        super(MomentExpansionApproximation, self).__init__(model)
        self.__n_moments = int(n_moments)

    def _wrapped_run(self):
        """
        Overrides the default _run() private method.
        Performs the complete analysis
        :return: an ODEProblem which can be further used in inference and simulation
        """

        n_moments = self.__n_moments
        stoichiometry_matrix = self.model.stoichiometry_matrix
        propensities = self.model.propensities
        species = self.model.species
        n_species = len(species)

        # compute n_counter and k_counter; the "n" and "k" vectors in equations, respectively.
        # n_counter = k_counter - first_order_moments
        n_counter, k_counter = self.generate_n_and_k_counters(n_moments, n_species)

        # Calculate TaylorExpansion terms to use in dmu/dt (eq. 6)
        taylor_expansion_matrix = taylor_expansion(species, propensities, n_counter)

        # dmu_over_dt is the product of the stoichiometry matrix by the Taylor Expansion terms.
        # one row per species and one col per element of n_counter
        dmu_over_dt = stoichiometry_matrix * taylor_expansion_matrix

        #  Calculate expressions to use in central moments equations (eq. 9)
        #  central_moments_exprs is a  matrix in which TODO  moment (n1,...,nd) combination.
        central_moments_exprs = eq_centralmoments(n_counter, k_counter, dmu_over_dt, species, propensities, stoichiometry_matrix)

        #  raw_to_central calculates central moments (symbolised by central_moments_symbols) in terms
        #  of raw moment expressions (raw_moment_exprs) (eq. 8)
        central_from_raw_exprs = raw_to_central(n_counter, species, k_counter)

        # Substitute raw moment, in central_moments, with of central moments
        central_moments_exprs = self.substitute_raw_with_central(central_moments_exprs, central_from_raw_exprs, n_counter, k_counter)

        # Get expressions for each central moment, and enter into list MFK
        MFK = self.make_mfk(central_moments_exprs, n_counter, dmu_over_dt)

        # concatenate the first order raw moments (means) and the
        # higher order central moments (variances, covariances,...)
        #the `reversed` is a hack to make the output similar to the original program
        prob_moments = [k for k in reversed(k_counter) if k.order == 1]
        prob_moments += [n for n in n_counter if n.order > 1]

        # problem_moment_nvecs = [tuple(pm.n_vector) for pm in prob_moments]
        # lhs = sp.Matrix([pm.symbol for pm in prob_moments])
        # prob_dic = dict(zip(lhs, problem_moment_nvecs))

        #TODO problem should use Moments in the constructor of ODEProblem
        out_problem = ode_problem.ODEProblem("MEA", prob_moments, MFK, sp.Matrix(self.model.constants))
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

        # Here we assume the same order. it would be better to ensure the moments n_vectors match
        # The symbols for raw moment symbols
        raw_lhs = [raw.symbol for raw in k_counter if raw.order > 1]
        # The symbols for the corresponding central moment
        central_symbols= [central.symbol for central in n_counter if central.order > 1]

        # Now we state (central_symbols - central_from_raw_exprs) == 0
        eq_to_solve = [cfr - cs for (cs, cfr) in zip(central_symbols, central_from_raw_exprs)]

        # And we solve this for the symbol of the corresponding raw moment
        # This gives an expression of the symbol for raw moment in terms of central moments
        # and lower order raw moment
        solved_xs = [sp.solve(rhs, rlhs) for (rhs, rlhs) in zip(eq_to_solve, raw_lhs)]

        out_exprs = central_moments_exprs.clone()
        # note the "reversed":
        # we start the substitutions by higher order moments and propagate to the lower order moments
        for (rlhs, sx) in reversed(zip(raw_lhs, solved_xs)):
            out_exprs = out_exprs.applyfunc(lambda x : sp.Subs(x, rlhs, sx).doit())

            #todo eventually, remove simplify (slow)
            out_exprs = out_exprs.applyfunc(sp.simplify)
        return out_exprs

    def make_mfk(self, central_moments, n_counter, dmu_over_dt):
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

    def generate_n_and_k_counters(self, n_moments,n_vars):
        """
        Makes a counter for raw moment (k_counter) and a counter for central moments (n_counter)
        Each is a list of "Moment" objects. Therefore, they are represented by both a vector of integer
        and a symbol.

        :param n_moments: the maximal order of moment to be computer
        :param n_vars: the number of variables
        :return: a pair of lists of Moments
        """
        #fixme clean this ugly function
        k_counter_tuples = [i for i in itertools.product(range(n_moments + 1), repeat=n_vars) if sum(i) <= n_moments]

        #fixme
        # substitute_raw_with_central assumes to have the lower order moment first
        #k_counter_tuples = sorted(k_counter_tuples, cmp=lambda x, y: sum(x) - sum(y))

        raw_symbols = [None] * len(k_counter_tuples)
        for i,count in enumerate(k_counter_tuples):
            if sum(count) == 0:
                raw_symbols[i] = sp.Integer(1)
            elif sum(count) == 1:
                idx = [j for j, c in enumerate(count) if c == 1][0]
                raw_symbols[i] = sp.Symbol("y_{0}".format(idx))
            else:
                raw_symbols[i] = sp.S("x_" + "_".join([str(s) for s in count]))


        k_counter = [Moment(c, s) for c,s in zip(k_counter_tuples, raw_symbols)]

        n_counter_tuples = [m for m in k_counter_tuples if sum(m) != 1]
        n_counter_symbols = [None] * len(n_counter_tuples)
        k = 0
        for i,count in enumerate(n_counter_tuples):
            if sum(count) == 0:
                n_counter_symbols[i] = sp.Integer(1)
            else:
                n_counter_symbols[i] = sp.S('yx{0}'.format(k+1))
                k += 1

        n_counter = [Moment(c, s) for c,s in zip(n_counter_tuples, n_counter_symbols)]

        return (n_counter, k_counter)

