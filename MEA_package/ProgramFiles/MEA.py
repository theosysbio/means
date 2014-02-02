####python MFK_final.py <model> <nMoments> <outputfile>

from time import time
import sys
import sympy as sp

from fcount import fcount
from TaylorExpansion import taylor_expansion
from centralmoments import eq_centralmoments
from model import parse_model
from raw_to_central import raw_to_central
from sympyhelpers import substitute_all
import ode_problem
from approximation_baseclass import ApproximationBaseClass


def substitute_mean_with_y(moments, n_species):
    """
    Replaces first order raw moments(e.g. x01, x10) by explicit means (e.g. y_0, y_1)

    :param moments: the list of expressions (moments)
    :param n_species: the number of species
    :return: the substituted expressions
    """

    diag_mat = [["1" if x == y else "0" for x in range(n_species)] for y in range(n_species)]
    substitutions_pairs = [('y_%i' % i, "x_" + "_".join(vec)) for (i,vec) in enumerate(diag_mat)]

    # for 2d lists
    if isinstance(moments[0], list):
        out_moms =[[substitute_all(m, substitutions_pairs) for m in mom ] for mom in moments]
    # 1d lists
    else:
        #out_moms =sp.Matrix([substitute_all(m, substitutions_pairs) for m in moments])
        out_moms = moments.applyfunc(lambda x: substitute_all(x, substitutions_pairs))

    return out_moms

def substitute_raw_with_central(central_moments, momvec, mom):
    #todo describe CentralMoments
    """
    Substitute raw moment terms in central_moments in terms of central moments
    (need to iterate in reverse from highest to lowest order moments to ensure all
    raw moments are replaced as some higher order raw moments are expressed in terms
    of central and lower order raw moments)

    :param central_moments: TODO
    :param momvec: the symbols for central moments (e.g. ym11, ym02, ...)
    :param mom:  the expressions of central moments in terms of raw moments
    :return: the substituted central moments
    """

    out_central_moments = central_moments.tolist()
    xs_to_solve = [sp.Symbol('x'+str(mv)[2:]) for mv in momvec]
    right_hand_sides = [m - mv for (mv, m) in zip(momvec, mom)]
    solved_xs = [sp.solve(rhs, xts) for (rhs, xts) in zip(right_hand_sides, xs_to_solve)]

    # note the "reversed":
    # we start the substitutions by higher order moments and propagate to the lower order moments
    for (xts, sx) in reversed(zip(xs_to_solve, solved_xs)):
        out_central_moments = [[sp.Subs(cm, xts, sx).doit()
                                    for cm in cent_mom] for cent_mom in out_central_moments]
        #todo eventualy, remove simplify (slow)
        out_central_moments = [[sp.simplify(cm) for cm in cent_mom] for cent_mom in out_central_moments]



    return sp.Matrix(out_central_moments)

def substitute_ym_with_yx(central_moments, momvec):
    """
    Substitute central moment terms ymn, where n gives n1,...nd combination
    with yxi where i indicates index in counter for that n1,...,nd

    :param CentralMoments:
    :param momvec: the symbols for central moments()
    :return: the symbols for central moments (e.g. yx1, yx2, ...)
    """
    yx_symbols = ['yx{0}'.format(i+1) for i in range(len(momvec))]
    # Any element in "momvec" should be replaced by yxN where N is its index (starting at one)

    #substitutions_pairs = [(yx, mom) for yx, mom in zip(momvec, yx_symbols)]
    substitutions_pairs = zip(yx_symbols,momvec)
    # apply this to all elements (in list and sub-list)
    #out_moms =sp.Matrix([[substitute_all(m, substitutions_pairs) for m in mom] for mom in central_moments.tolist()])
    out_moms = central_moments.applyfunc(lambda x: substitute_all(x, substitutions_pairs))

    return (yx_symbols, out_moms)


def make_mfk(central_moments , yms, M):
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
    return MFK

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
        S = self.model.stoichiometry_matrix
        amat = self.model.propensities

        ymat = self.model.species
        n_species = len(ymat)

        # compute counter and mcounter; the "k" and "n" vectors in equations. counter = mcounter - first_order_moments
        (counter, mcounter) = fcount(n_moments, n_species)
        # Calculate TaylorExpansion terms to use in dmu/dt (eq. 6)
        TE_matrix = taylor_expansion(ymat, amat, counter)

        # M is the product of the stoichiometry matrix by the Taylor Expansion terms.
        # one row per species and one col per element of counter
        M = S * TE_matrix

        #  Calculate expressions to use in central moments equations (eq. 9)
        #  CentralMoments is a list with entry for each moment (n1,...,nd) combination.
        central_moments = eq_centralmoments(counter, mcounter, M, ymat, amat, S)

        #  Substitute means in CentralMoments by y_i (ymat entry)
        central_moments = substitute_mean_with_y(central_moments, n_species)

        #  Substitute higher order raw moments in terms of central moments
        #  raw_to_central calculates central moments (momvec) in terms
        #  of raw moment expressions (mom) (eq. 8)
        (mom, momvec) = raw_to_central(counter, ymat, mcounter)

        # Substitute one for zeroth order raw moments in mom
        x_zero = sp.Symbol("x_" + "_".join(["0"] * n_species))

        mom = mom.applyfunc(lambda x : sp.Subs(x, x_zero, sp.S(1)).doit() )

        # Substitute first order raw moments (means) in mom with y_i (ymat entry)
        mom = substitute_mean_with_y(mom, n_species)

        # Substitute raw moment, in central_moments, with of central moments
        central_moments = substitute_raw_with_central(central_moments, momvec, mom)


        # Use symbols for central moments (ymn) as yxN where N is a counter from one (e.g. ym_0_0_2 -> yx1)
        yx_symbols, central_moments = substitute_ym_with_yx(central_moments, momvec)


        # Make yms; (yx1, yx2, yx3,...,yxn) where n is the number of elements in counter

        # Set zeroth order central moment to 1
        yms = sp.Matrix([sp.S("1")] + yx_symbols)

        # Get expressions for each central moment, and enter into list MFK
        MFK = make_mfk(central_moments, yms, M)

        # build ODEProblem object
        prob_moments = [tuple([1 if i==j else 0 for i in range(n_species)]) for j in range(n_species)]
        prob_moments += [tuple(c) for c in counter[1:]]

        lhs = sp.Matrix([i for i in ymat] + yms[1:])

        prob_moments = dict(zip(lhs,prob_moments))

        out_problem = ode_problem.ODEProblem("MEA", lhs, sp.Matrix(MFK), sp.Matrix(self.model.constants), prob_moments)
        return out_problem

def get_args():
    model_ = sys.argv[1]
    numMoments = int(sys.argv[2])
    out_file_name = str(sys.argv[3])
    if numMoments < 2:
        raise ValueError("The number of moments (--nMom) must be greater than one")

    return (model_, numMoments, out_file_name)

if __name__ == "__main__":

    # get and validate command line arguments
    model_filename, n_moments, out_file_name = get_args()

    # parse the input file as a Model object
    model = parse_model(model_filename)

    # set the mea analysis up
    mea = MomentExpansionApproximation(model, n_moments)

    # run mea with the defined parameters
    problem = mea.run()

    # write result in the specified file
    ode_writer = ode_problem.ODEProblemWriter(problem, mea.time_last_run)
    ode_writer.write_to(out_file_name)
    tex_writer = ode_problem.ODEProblemLatexWriter(problem)
    tex_writer.write_to(out_file_name + ".tex")




