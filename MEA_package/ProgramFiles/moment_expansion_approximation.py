import itertools
import sympy as sp
from approximation_baseclass import ApproximationBaseClass
from sympyhelpers import substitute_all
import ode_problem
from ode_problem import Moment

from TaylorExpansion import taylor_expansion
from centralmoments import eq_centralmoments
from raw_to_central import raw_to_central

class MomentExpansionApproximation(ApproximationBaseClass):

    """
    Performs moment expansion approximation (Ale et al. 2013) up to a given order of moment.
    """

    def __init__(self, model, n_moments):
        super(MomentExpansionApproximation, self).__init__(model)
        self.__n_moments = n_moments

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

        # compute counter and mcounter; the "k" and "n" vectors in equations. counter = mcounter - first_order_moments
        (counter, mcounter) = self.fcount(n_moments, n_species)

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
        (central_from_raw_exprs, used_moments) = raw_to_central(counter, species, mcounter)

        # Substitute raw moment, in central_moments, with of central moments
        central_moments_exprs = self.substitute_raw_with_central(central_moments_exprs, used_moments, central_from_raw_exprs)


        yx_symbols = [i.central_symbol for i in used_moments]
        # prepend with zeroth order central moment (which is one)
        yx_symbols = sp.Matrix([sp.Integer("1")] + yx_symbols)
        # Get expressions for each central moment, and enter into list MFK
        MFK = self.make_mfk(central_moments_exprs, yx_symbols, M)

        # build ODEProblem object
        prob_moments = [tuple([1 if i==j else 0 for i in range(n_species)]) for j in range(n_species)]
        prob_moments += [tuple(c.n_vector) for c in counter[1:]]

        #symbols for the left hand side equations.. first order raw moments followed by higher order moments
        lhs = sp.Matrix([i for i in species] + yx_symbols[1:])

        prob_moments = dict(zip(lhs,prob_moments))

        out_problem = ode_problem.ODEProblem("MEA", lhs, MFK, sp.Matrix(self.model.constants), prob_moments)
        return out_problem

    def substitute_raw_with_central(self, central_moments_exprs, used_moments, central_from_raw_exprs):
        """
        Substitute raw moment terms in central_moments in terms of central moments
        (need to iterate in reverse from highest to lowest order moments to ensure all
        raw moments are replaced as some higher order raw moments are expressed in terms
        of central and lower order raw moments)
        :param central_moments_exprs: a matrix of expressions for central moments.
        :param central_moments_symbols: the symbols for central moments (e.g. ym_1_1, ym_0_2, ...)
        :param central_from_raw_exprs:  the expressions of central moments in terms of raw moments
        :return: the substituted central moments
        """

        xs_to_solve = [um.raw_symbol for um in used_moments]
        right_hand_sides = [m - um.central_symbol for (um, m) in zip(used_moments, central_from_raw_exprs)]
        solved_xs = [sp.solve(rhs, xts) for (rhs, xts) in zip(right_hand_sides, xs_to_solve)]

        # note the "reversed":
        # we start the substitutions by higher order moments and propagate to the lower order moments
        out_exprs = central_moments_exprs.clone()
        for (xts, sx) in reversed(zip(xs_to_solve, solved_xs)):
            out_exprs = out_exprs.applyfunc(lambda x : sp.Subs(x, xts, sx).doit())
            #todo eventually, remove simplify (slow)
            out_exprs = out_exprs.applyfunc(sp.simplify)
        return out_exprs

    def make_mfk(self, central_moments , yms, M):
        """
        :param CentralMoments:
        :param yms:
        :param M:
        :return: MFK ...
        """

        # try to simplify an expression. returns the original expression if fail
        # todo remove this when we do not need it anymore
        def try_to_simplify(expr):
            try:
                return sp.simplify(expr)
            except:
                pass
            return expr

        # todo eventually, we want to remove the simplify calls#
        MFK = [try_to_simplify(e) for e in M*yms ]
        MFK += [try_to_simplify((sp.Matrix(cm).T * yms)[0]) for cm in central_moments.tolist()]
        return sp.Matrix(MFK)


    def fcount(self, n_moments,n_vars):
        """
        :param n_moments: the maximal order of moment to be computer
        :param n_vars: the number of variables
        :return: a pair of tuples. the first element contains the all the permutations,
        whilst the second element does not have the first order (e.g. {0,0,1})
        """
        #todo discus the new status of Moment, counter...


        m_counter = [i for i in itertools.product(range(n_moments + 1), repeat=n_vars) if sum(i) <= n_moments]

        m_counter = sorted(m_counter, cmp=lambda x, y: sum(x) - sum(y))
        # build symbols for raw moments
        raw_symbols = [None] * len(m_counter)
        central_symbols = [None] * len(m_counter)
        k = 0

        for i,count in enumerate(m_counter):
            if sum(count) == 0:
                raw_symbols[i] = sp.Integer(1)
            elif sum(count) == 1:
                idx = [j for j, c in enumerate(count) if c == 1][0]
                raw_symbols[i] = sp.Symbol("y_{0}".format(idx))
            else:
                raw_symbols[i] = sp.S("x_" + "_".join([str(s) for s in count]))
                central_symbols[i] = sp.S('yx{0}'.format(k+1))
                k += 1

        m_counter = [Moment(c, rs, cs) for c,rs,cs in zip(m_counter, raw_symbols, central_symbols)]

        counter = [i for i in m_counter if i.order != 1]
        return (counter, m_counter)

