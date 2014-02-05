import itertools

import sympy as sp

from means.approximation import ode_problem
from means.approximation.approximation_baseclass import ApproximationBaseClass
from means.approximation.ode_problem import Moment
from TaylorExpansion import taylor_expansion
from centralmoments import eq_centralmoments
from raw_to_central import raw_to_central


class MomentExpansionApproximation(ApproximationBaseClass):

    """
    Performs moment expansion approximation (Ale et al. 2013) up to a given order of moment.
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

        # compute counter and mcounter; the "n" and "k" vectors in equations, respectively.
        # counter = mcounter - first_order_moments
        counter, mcounter = self.fcount(n_moments, n_species)

        # Calculate TaylorExpansion terms to use in dmu/dt (eq. 6)
        taylor_expansion_matrix = taylor_expansion(species, propensities, counter)

        # M is the product of the stoichiometry matrix by the Taylor Expansion terms.
        # one row per species and one col per element of counter
        #todo find a name and description for M
        M = stoichiometry_matrix * taylor_expansion_matrix

        #  Calculate expressions to use in central moments equations (eq. 9)
        #  central_moments_exprs is a  matrix in which TODO  moment (n1,...,nd) combination.
        central_moments_exprs = eq_centralmoments(counter, mcounter, M, species, propensities, stoichiometry_matrix)

        #  raw_to_central calculates central moments (symbolised by central_moments_symbols) in terms
        #  of raw moment expressions (raw_moment_exprs) (eq. 8)
        central_from_raw_exprs = raw_to_central(counter, species, mcounter)

        # Substitute raw moment, in central_moments, with of central moments
        central_moments_exprs = self.substitute_raw_with_central(central_moments_exprs, central_from_raw_exprs, counter, mcounter)

        # Get expressions for each central moment, and enter into list MFK
        MFK = self.make_mfk(central_moments_exprs, counter, M)

        # concatenate the first order raw moments (means) and the
        # higher order central moments (variances, covariances,...)
        #the `reversed` is a hack to make the output similar to the original program
        prob_moments = [m for m in reversed(mcounter) if m.order == 1]
        prob_moments += [c for c in counter if c.order > 1]

        # problem_moment_nvecs = [tuple(pm.n_vector) for pm in prob_moments]
        # lhs = sp.Matrix([pm.symbol for pm in prob_moments])
        # prob_dic = dict(zip(lhs, problem_moment_nvecs))

        #TODO problem should use Moments in the constructor of ODEProblem
        out_problem = ode_problem.ODEProblem("MEA", prob_moments, MFK, sp.Matrix(self.model.constants))
        return out_problem

    def substitute_raw_with_central(self, central_moments_exprs, central_from_raw_exprs, counter, mcounter):
        """
        Takes the expressions for central moments, and substitute the symbols representing raw moments,
        by equivalent expressions in terms of central moment
        :param central_moments_exprs: a matrix of expressions for central moments.
        :param central_from_raw_exprs: central moment expressed in terms of raw moments
        :param counter:
        :param mcounter:
        :return: expression of central moments without raw moment
        """

        # here we assume the same order. it would be better to ensure the moments n_vectors match
        # The symbols for raw moment symbols
        raw_lhs = [um.symbol for um in mcounter if um.order > 1]
        # The symbols for the corresponding central moment
        central_symbols= [um.symbol for um in counter if um.order > 1]

        # Now we state (central_symbols - central_from_raw_exprs) == 0
        eq_to_solve = [m - cs for (cs, m) in zip(central_symbols, central_from_raw_exprs)]

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

    def make_mfk(self, central_moments, counter, M):
        """
        :param central_moments:
        :param counter:
        :param M:
        :return: MFK ...
        """

        yms = sp.Matrix([c.symbol for c in counter])
        # try to simplify an expression. returns the original expression if fail
        # todo remove this when we do not need it anymore
        def try_to_simplify(expr):
            try:
                return sp.simplify(expr)
            except:
                pass
            return expr

        # todo eventually, we want to remove the simplify calls#
        MFK = [try_to_simplify(e) for e in M*yms]
        MFK += [try_to_simplify((sp.Matrix(cm).T * yms)[0]) for cm in central_moments.tolist()]
        return sp.Matrix(MFK)

    def fcount(self, n_moments,n_vars):
        """
        Makes a counter for raw moment (mcounter) and a counter for central moments (counter)
        Each is a list of "Moment" objects. Therefore, they are represented by both a vector of integer
        and a symbol.

        :param n_moments: the maximal order of moment to be computer
        :param n_vars: the number of variables
        :return: a pair of lists of Moments
        """
        #fixme clean this ugly function
        m_counter_tuples = [i for i in itertools.product(range(n_moments + 1), repeat=n_vars) if sum(i) <= n_moments]
        #m_counter_tuples = sorted(m_counter_tuples, cmp=lambda x, y: sum(x) - sum(y))

        raw_symbols = [None] * len(m_counter_tuples)
        for i,count in enumerate(m_counter_tuples):
            if sum(count) == 0:
                raw_symbols[i] = sp.Integer(1)
            elif sum(count) == 1:
                idx = [j for j, c in enumerate(count) if c == 1][0]
                raw_symbols[i] = sp.Symbol("y_{0}".format(idx))
            else:
                raw_symbols[i] = sp.S("x_" + "_".join([str(s) for s in count]))


        m_counter = [Moment(c, s) for c,s in zip(m_counter_tuples, raw_symbols)]

        counter_tuples = [m for m in m_counter_tuples if sum(m) != 1]
        counter_symbols = [None] * len(counter_tuples)
        k = 0
        for i,count in enumerate(counter_tuples):
            if sum(count) == 0:
                counter_symbols[i] = sp.Integer(1)
            else:
                counter_symbols[i] = sp.S('yx{0}'.format(k+1))
                k += 1

        counter = [Moment(c, s) for c,s in zip(counter_tuples, counter_symbols)]

        return (counter, m_counter)

