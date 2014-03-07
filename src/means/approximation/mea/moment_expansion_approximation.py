import itertools
import sympy as sp

from means.approximation.ode_problem import ODEProblem
from means.approximation.approximation_baseclass import ApproximationBaseClass
from means.approximation.ode_problem import Moment

# helper functions
from dmu_over_dt import generate_dmu_over_dt
from eq_central_moments import eq_central_moments
from raw_to_central import raw_to_central
from means.util.sympyhelpers import substitute_all, quick_solve

# the different closure methods:
from closure_gamma import GammaClosure
from closure_log_normal import LogNormalClosure
from closure_normal import NormalClosure
from closure_scalar import ScalarClosure


def mea_approximation(model, max_order, closure='scalar', *closure_args, **closure_kwargs):
    r"""
    A wrapper around :class:`~means.approximation.mea.moment_expansion_approximation.MomentExpansionApproximation`.
    It performs moment expansion approximation (MEA) as described in [Ale2013]_ up to a given order of moment.
    See :class:`~means.approximation.mea.moment_expansion_approximation.MomentExpansionApproximation` for details
    about the options.


    :return: an ODE problem which can be further used in inference and simulation.
    :rtype: :class:`~means.approximation.ode_problem.ODEProblem`
    """
    mea = MomentExpansionApproximation(model, max_order, closure=closure, *closure_args, **closure_kwargs)
    return mea.run()


class MomentExpansionApproximation(ApproximationBaseClass):
    r"""
    A class to perform moment expansion approximation as described in [Ale2013]_ up to a given order of moment.
    In addition, it allows to close the Taylor expansion by using parametric values for last order central moments.

    .. [Ale2013] A. Ale, P. Kirk, and M. P. H. Stumpf,\
    "A general moment expansion method for stochastic kinetic models,"\
     The Journal of Chemical Physics, vol. 138, no. 17, p. 174101, 2013.

    """
    def __init__(self, model, max_order, closure='scalar', *closure_args, **closure_kwargs):

        r"""
        :param model: The model to be approximated
        :type model: :class:`~means.model.model.Model`

        :param max_order: the highest order of central moments in the resulting ODEs
        :param closure: a string describing the type of closure to use. Currently, the supported closures are:

            `'scalar'`
                higher order central moments are set to zero.
                See :class:`~means.approximation.mea.closure_scalar.ScalarClosure`.
            `'normal'`
                uses normal distribution to compute last order central moments.
                See :class:`~means.approximation.mea.closure_normal.NormalClosure`.
            `'log-normal'`
                uses log-normal distribution.
                See :class:`~means.approximation.mea.closure_log_normal.LogNormalClosure`.
            `'gamma'`
                EXPERIMENTAL,
                uses gamma distribution.
                See :class:`~means.approximation.mea.closure_gamma.GammaClosure`.

        :type closure: string
        :param closure_args: arguments to be passed to the closure
        :param closure_kwargs: keyword arguments to be passed to the closure
        """
        super(MomentExpansionApproximation, self).__init__(model)
        try:
            self.__max_order = int(max_order)
            if self.__max_order < 1:
                raise ValueError("`max_order` can only be POSITIVE")
        except:
            raise ValueError("`max_order` can only be positive integer")

        # A dictionary of "option -> closure". this allows a generic handling for closure without having to add
        # if-else and exceptions when implementing new closures. One only needs to add the new closure class to the dict
        supported_closures = {"log-normal": LogNormalClosure,
                             "scalar": ScalarClosure,
                             "normal": NormalClosure,
                             "gamma": GammaClosure}

        # We initialise the closure for this approximator
        try:
            # our closure is an instance of the class queried in the dictionary
            ClosureClass = supported_closures[closure]
            self.__closure = ClosureClass(self.__max_order, *closure_args, **closure_kwargs)
        except KeyError:
            error_str = "The closure type '{0}' is not supported.\n\
                         Supported values for closure:\
                         {1}"
            raise KeyError(error_str.format(closure, supported_closures))

    @property
    def closure(self):
        return self.__closure

    def run(self):
        r"""
        Overrides the default run() method.
        Performs the complete analysis on the model specified during initialisation.

        :return: an ODE problem which can be further used in inference and simulation.
        :rtype: :class:`~means.approximation.ode_problem.ODEProblem`
        """
        max_order = self.__max_order
        stoichiometry_matrix = self.model.stoichiometry_matrix
        propensities = self.model.propensities
        species = self.model.species
        # compute n_counter and k_counter; the "n" and "k" vectors in equations, respectively.
        n_counter, k_counter = self._generate_n_and_k_counters(max_order, species)
        # dmu_over_dt has row per species and one col per element of n_counter (eq. 6)
        dmu_over_dt = generate_dmu_over_dt(species, propensities, n_counter, stoichiometry_matrix)
        # Calculate expressions to use in central moments equations (eq. 9)
        central_moments_exprs = eq_central_moments(n_counter, k_counter, dmu_over_dt, species, propensities, stoichiometry_matrix, max_order)
        # Expresses central moments in terms of raw moments (and central moments) (eq. 8)
        central_from_raw_exprs = raw_to_central(n_counter, species, k_counter)
        # Substitute raw moment, in central_moments, with expressions depending only on central moments
        central_moments_exprs = self._substitute_raw_with_central(central_moments_exprs, central_from_raw_exprs, n_counter, k_counter)
        # Get final right hand side expressions for each moment in a vector
        mfk = self._generate_mass_fluctuation_kinetics(central_moments_exprs, dmu_over_dt, n_counter)
        # Applies moment expansion closure, that is replaces last order central moments by parametric expressions
        mfk = self.closure.close(mfk, central_from_raw_exprs, n_counter, k_counter)
        # These are the left hand sign symbols referring to the mfk
        prob_lhs = self._generate_problem_left_hand_side(n_counter, k_counter)
        # Finally, we build the problem
        out_problem = ODEProblem("MEA", prob_lhs, mfk, sp.Matrix(self.model.constants))
        return out_problem

    def _generate_problem_left_hand_side(self, n_counter, k_counter):
        """
        Generate the left hand side of the ODEs. This is simply the symbols for the corresponding moments.
        Note that, in principle, they are in of course fact the time derivative of the moments.

        :param n_counter: a list of :class:`~means.approximation.ode_problem.Moment`\s representing central moments
        :param k_counter: a list of :class:`~means.approximation.ode_problem.Moment`\s representing raw moments
        :return: a list of the problem left hand sides
        :rtype: list[:class:`sympy.Symbol`]
        """

        # concatenate the symbols for first order raw moments (means)
        prob_moments_over_dt = [k for k in k_counter if k.order == 1]
        # and the higher order central moments (variances, covariances,...)
        prob_moments_over_dt += [n for n in n_counter if self.__max_order >= n.order > 1]


        return prob_moments_over_dt

    def _generate_mass_fluctuation_kinetics(self, central_moments, dmu_over_dt, n_counter):
        """
        Generate the Mass Fluctuation Kinetics (i.e. the right hand side of the ODEs)

        :param central_moments:
        :param dmu_over_dt:
        :param n_counter: a list of :class:`~means.approximation.ode_problem.Moment`\s representing central moments

        :return: the MFK as a matrix
        :rtype: :class:`sympy.Matrix`
        """

        # symbols for central moments
        central_moments_symbols = sp.Matrix([n.symbol for n in n_counter])

        # rhs for the first order raw moment
        mfk = [e for e in dmu_over_dt * central_moments_symbols]
        # rhs for the higher order raw moments
        mfk += [(sp.Matrix(cm).T * central_moments_symbols)[0] for cm in central_moments.tolist()]

        mfk = sp.Matrix(mfk)

        return mfk

    def _substitute_raw_with_central(self, central_moments_exprs, central_from_raw_exprs, n_counter, k_counter):
        r"""
        Takes the expressions for central moments, and substitute the symbols representing raw moments,
        by equivalent expressions in terms of central moment

        :param central_moments_exprs: a matrix of expressions for central moments.
        :param central_from_raw_exprs: central moment expressed in terms of raw moments
        :param n_counter: a list of :class:`~means.approximation.ode_problem.Moment`\s representing central moments
        :type n_counter: list[:class:`~means.approximation.ode_problem.Moment`]
        :param k_counter: a list of :class:`~means.approximation.ode_problem.Moment`\s representing raw moments
        :type k_counter: list[:class:`~means.approximation.ode_problem.Moment`]
        :return: expressions for central moments without raw moment
        """
        positiv_raw_moms_symbs = [raw.symbol for raw in k_counter if raw.order > 1]
        # The symbols for the corresponding central moment
        central_symbols= [central.symbol for central in n_counter if central.order > 1]
        # Now we state (central_symbols == central_from_raw_exprs)
        eq_to_solve = [cfr - cs for (cs, cfr) in zip(central_symbols, central_from_raw_exprs)]

        # And we solve this for the symbol of the corresponding raw moment. This gives an expression
        # of the symbol for raw moment in terms of central moments and lower order raw moment
        solved_xs = sp.Matrix([quick_solve(eq,raw) for (eq, raw) in zip(eq_to_solve, positiv_raw_moms_symbs)])

        # now we want to express raw moments only in terms od central moments and means
        # for instance if we have: :math:`x_1 = 1; x_2 = 2 +x_1 and  x_3 = x_2*x_1`, we should give:
        # :math:`x_1 = 1; x_2 = 2+1 and  x_3 = 1*(2+1)`
        # To achieve this, we recursively apply substitution as many times as the highest order (minus one)
        max_order = max([p.order for p in k_counter])

        for i in range(max_order - 1):
            substitution_pairs = zip(positiv_raw_moms_symbs, solved_xs)
            solved_xs = substitute_all(solved_xs, substitution_pairs)

        # we finally build substitution pairs to replace all raw moments
        substitution_pairs = zip(positiv_raw_moms_symbs, solved_xs)

        # apply this substitution to all elements of the central moment expressions matrix
        out_exprs = substitute_all(central_moments_exprs, substitution_pairs)

        return out_exprs

    def _generate_n_and_k_counters(self, max_order, species, central_symbols_prefix="yx", raw_symbols_prefix="x_"):
        r"""
        Makes a counter for central moments (n_counter) and a counter for raw moment (k_counter).
        Each is a list of :class:`~means.approximation.ode_problem.Moment`s.
        Therefore, each :class:`~means.approximation.ode_problem.Moments` is represented by both
        a vector of integer and a symbol.

        :param max_order: the maximal order of moment to be computer
        :param species: the name of the species
        :return: a pair of lists of :class:`~means.approximation.ode_problem.Moment`s
        """
        n_moments = max_order + 1
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

        #this mimics the order in the original code
        k_counter_descriptors = sorted(k_counter_descriptors,lambda x,y: sum(x) - sum(y))
        #k_counter_descriptors = [[r for r in reversed(k)] for k in k_counter_descriptors]
        k_counter_symbols = [sp.Symbol(raw_symbols_prefix + "_".join([str(s) for s in count]))
                             for count in k_counter_descriptors]
        k_counter += [Moment(d, s) for d,s in zip(k_counter_descriptors, k_counter_symbols)]

        #  central moments
        n_counter_descriptors = [m for m in k_counter_descriptors if sum(m) > 1]
        # arbitrary symbols
        n_counter_symbols = [sp.Symbol(central_symbols_prefix + str(i+1)) for i in range(len(n_counter_descriptors))]
        n_counter += [Moment(c, s) for c,s in zip(n_counter_descriptors, n_counter_symbols)]

        return n_counter, k_counter

