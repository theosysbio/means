####python MFK_final.py <model> <nMoments> <outputfile>

from time import time
import sys
import itertools

from fcount import fcount
import sympy as sp
from sympy import Matrix, var
from TaylorExpansion import taylor_expansion
from centralmoments import eq_centralmoments
from model import parse_model
from raw_to_central import raw_to_central
from sympy import latex
from sympyhelpers import substitute_all
import ode_problem
def substitute_mean_with_y(moments, nvariables):
    """
    Replaces first order raw moments(e.g. x01, x10) by explicit means (e.g. y_0, y_1)

    :param moments: the list of expressions (moments)
    :param nvariables: the number of species
    :return: the substituted expressions
    """

    diag_mat = [["1" if x == y else "0" for x in range(nvariables)] for y in range(nvariables)]
    substitutions_pairs = [('y_%i' % i, "x_" + "_".join(vec)) for (i,vec) in enumerate(diag_mat)]

    # for 2d lists
    if isinstance(moments[0], list):
        out_moms =[[substitute_all(m, substitutions_pairs) for m in mom ] for mom in moments]
    # 1d lists
    else:
        out_moms =[substitute_all(m, substitutions_pairs) for m in moments]

    return out_moms

def substitute_raw_with_central(CentralMoments, momvec, mom):
    #todo describe CentralMoments
    """
    Substitute raw moment terms in CentralMoments in terms of central moments
    (need to iterate in reverse from highest to lowest order moments to ensure all
    raw moments are replaced as some higher order raw moments are expressed in terms
    of central and lower order raw moments)

    :param CentralMoments: TODO
    :param momvec: the symbols for central moments (e.g. ym11, ym02, ...)
    :param mom:  the expressions of central moments in terms of raw moments
    :return: the substituted central moments
    """

    out_central_moments = CentralMoments[:]

    #x_to_solve =)

    xs_to_solve = [sp.Symbol('x'+str(mv)[2:]) for mv in momvec]
    right_hand_sides = [m - mv for (mv, m) in zip(momvec, mom)]
    solved_xs = [sp.solve(rhs, xts) for (rhs, xts) in zip(right_hand_sides, xs_to_solve)]


    for (xts, sx) in reversed(zip(xs_to_solve, solved_xs)):
        out_central_moments = [[sp.Subs(cm, xts, sx).doit() for cm in cent_mom] for cent_mom in out_central_moments]
#
        out_central_moments = [[sp.simplify(cm) for cm in cent_mom] for cent_mom in out_central_moments]



    return out_central_moments

def substitute_ym_with_yx(central_moments, momvec):
    """
    Substitute central moment terms ymn, where n gives n1,...nd combination
    with yxi where i indicates index in counter for that n1,...,nd

    :param CentralMoments:
    :param momvec: the symbols for central moments ()
    :return: the symbols for central moments (e.g. yx1, yx2, ...)
    """

    # Any element in "momvec" should be replaced by yxN where N is its index (starting at one)
    substitutions_pairs = [('yx%i' % (i + 1), mom) for i, mom in enumerate(momvec)]

    # apply this to all elements
    out_moms =[[substitute_all(m, substitutions_pairs) for m in mom] for mom in central_moments]

    return out_moms

def make_mfk(CentralMoments, yms, M):
    #TODO figure-out what MFK stands for
    """
    :param CentralMoments:
    :param yms:
    :param M:
    :return: MFK ...
    """

######################################
    # this is the new code without simplify
    # this make the tests fail for simulation because of floating point issues.
    # Implementation postponed

    # MFK =  [i for i in M*yms ]
    # MFK += [(sp.Matrix(cm).T * yms)[0] for cm in CentralMoments]
    # return MFK
#####################################
    # Get expressions for higher order central moments
    MFK1 = M*yms



    MFK = []
    # Reshape to a vector
    for i in range(0,len(MFK1)):
        try:
            MFK1[i] = sp.simplify(MFK1[i])
            #MFK1[i] = sp.collect(sp.expand(MFK1[i]),yms)
        except:
            pass
        MFK.append(MFK1[i])



    for i in range(0,len(CentralMoments)):
        rowmat = sp.Matrix(1, len(CentralMoments[i]), CentralMoments[i])

        # This should be a scalar
        MFK2 = rowmat * yms
        # TODO scalar => {len() == 1} so why do we need a loop here ?
        for j in range(0,len(MFK2)):
            try:
                MFK2[j] = sp.simplify(MFK2[j])
                #MFK2[i] = sp.collect(sp.expand(MFK2[i]),yms)
            except:
                pass
        MFK.append(MFK2[0])

    return MFK

def write_output(out_file_prefix, nvariables, nMoments, counter, c, yms, ymat, MFK, deltatime):

    """
    Write output and latex formatted equations

    :param out_file_prefix: name of the output file. Default is ODEout.
    :param nvariables: number of species
    :param nMoments: the number of moments used in expansion
    :param counter: the combination of orders of derivation
    :param c: the constants. Provided by the model
    :param yms: the vector of symbols for the central moments
    :param ymat:
    :param MFK:
    :param deltatime: the elapsed time
    """


    output = open(out_file_prefix,'w')
    # Create list with moment names (moment_list)
    moment_list = []
    for i in range(0, nvariables):
        means_vec = [0]*nvariables
        means_vec[i] = 1
        moment_list.append(means_vec)
    moment_list += counter[1:]


    # Get list of parameters for LHS of ODEs

    LHS = ymat[:]
    for moms in range(1,len(yms)):
        LHS.append(yms[moms])

    constants = c[:]

    out_tex = open(out_file_prefix+'.tex','w')
    out_tex.write('\documentclass{article}\n\usepackage[landscape, margin=0.5in, a3paper]{geometry}\n\\begin{document}\n\section*{RHS of equations}\n')

    # Write equations, LHS, constants, moment names, and numbers of variables/moments/equations

    output.write('MEA\n\nRHS of equations:\n')
    for i in range(len(MFK)):
            output.write(str(MFK[i])+'\n')
            out_tex.write('$\dot '+str(latex(LHS[i]))+ ' = '+str(latex(MFK[i]))+'$\\\\\\\\')
    output.write('\nLHS:\n')
    out_tex.write('\n\section*{Moments}\n')
    for i in range(len(LHS)):
        output.write(str(LHS[i])+'\n')
        out_tex.write('\n$'+str(latex(LHS[i]))+'$: {'+str(moment_list[i])+'}\\\\')

    output.write('\nConstants:\n')
    for i in range(len(constants)):
        output.write(str(constants[i])+'\n')

    output.write('Number of variables:\n'+str(nvariables)+'\nNumber of moments:\n'+str(nMoments)+'\nTime taken (s): '+str(deltatime))

    output.write('\n\nNumber of equations:\n'+str(len(LHS)))
    output.write('\n\nList of moments:\n')
    for i in range(len(moment_list)):
        output.write(str(moment_list[i])+'\n')

    output.close()

    out_tex.write('\n\end{document}')
    out_tex.close()

def MFK_final(model_filename, nMoments):

    """
    Produces central moment equations using the specified up to a given order.
    :param model_filename: file that contains model information
    :param nMoments: the number of moments used in expansion
    """

    # Set the timer (in order to report how long the execution of this function took)
    time1 = time()

    model = parse_model(model_filename)

    # TODO: make the terms pythonic
    S = model.stoichiometry_matrix
    amat = model.propensities
    ##todo unused
    nreactions = model.number_of_reactions
    nvariables = model.number_of_species
    ymat = model.species

    c = model.constants

    # compute counter and mcounter; the "k" and "n" vectors in equations. counter = mcounter - first_order_moments
    (counter, mcounter) = fcount(nMoments, nvariables)
    # Calculate TaylorExpansion terms to use in dmu/dt (eq. 6)
    TE_matrix = taylor_expansion(ymat, amat, counter)

    # M is the product of the stoichiometry matrix by the Taylor Expansion terms.
    # one row per species and one col per element of counter
    M = S * TE_matrix

    #  Calculate expressions to use in central moments equations (eq. 9)
    #  CentralMoments is a list with entry for each moment (n1,...,nd) combination.
    central_moments = eq_centralmoments(counter, mcounter, M, ymat, amat, S)
    #  Substitute means in CentralMoments by y_i (ymat entry)

    central_moments = substitute_mean_with_y(central_moments, nvariables)


    #  Substitute higher order raw moments in terms of central moments
    #  raw_to_central calculates central moments (momvec) in terms
    #  of raw moment expressions (mom) (eq. 8)
    (mom, momvec) = raw_to_central(counter, ymat, mcounter)

    # Substitute one for zeroth order raw moments in mom
    symbol_one = sp.S(1)
    x_zero = sp.Symbol("x_" + "_".join(["0"] * nvariables))
    mom = [sp.Subs(m, x_zero, symbol_one).doit() for m in mom]


    # Substitute first order raw moments (means) in mom with y_i (ymat entry)
    mom = substitute_mean_with_y(mom,nvariables)

    # Substitute raw moment, in CentralMoments, with of central moments
    central_moments = substitute_raw_with_central(central_moments, momvec, mom)


    # Use counter index (c) for yx (yxc) instead of moment (ymn) (e.g. ym021)
    central_moments = substitute_ym_with_yx(central_moments, momvec)


    # Make yms; (yx1, yx2, yx3,...,yxn) where n is the number of elements in counter
    if len(central_moments) != 0:
        nM = len(central_moments[0])
    else:
        nM = 1

    yms = sp.Matrix(nM, 1, lambda i, j : var('yx%i' % i))
    # Set zeroth order central moment to 1
    yms[0] = 1

    # Get expressions for each central moment, and enter into list MFK
    MFK = make_mfk(central_moments, yms, M)





    # Write information to output file (and moment names and equations to .tex file)
    write_output(str(sys.argv[3]), nvariables, nMoments, counter, c, yms, ymat, MFK, time() - time1)

    #todo use dedicated writer
    # prob_moments = [tuple([1 if i==j else 0 for i in range(nvariables) ]) for j in range(nvariables)]
    # prob_moments += [tuple(c) for c in counter[1:]]
    # lhs = sp.Matrix([i for i in ymat] + yms[1:])
    # problem = ode_problem.ODEProblem("MEA", lhs , sp.Matrix(MFK), sp.Matrix(c), prob_moments)
    # ode_problem.ODEProblem_writer(problem).write_to(str(sys.argv[3]))


def get_args():
    model_ = sys.argv[1]
    numMoments = int(sys.argv[2])

    if numMoments < 2:
        raise ValueError("The number of moments (--nMom) must be greater than one")

    return (model_, numMoments)


if __name__ == "__main__":
    model_, numMoments = get_args()
    MFK_final(model_, numMoments)
