####python MFK_final.py <model> <nMoments> <outputfile>

from time import time
import sys
import os

from sympy import Matrix, diff, Symbol, Subs, Eq, var, simplify
from fcount import fcount
from sympy import Matrix, diff, Symbol, Subs, Eq, var, simplify
from sympy import S as F
from sympy.solvers import solve
from TaylorExpansion import taylor_expansion
from centralmoments import eq_centralmoments
from formatmodel import parse_model
from raw_to_central import raw_to_central
from sympy import latex
import sympy

def get_and_check_model(model_filename):

    S,a,nreactions,nvariables,ymat,Mumat, c = parse_model(model_filename).legacy_interface()

    #Delete temporary model file
    #TODO use assertions instead of `if else`
    #Check stoichiometry matrix
    if S.cols==nreactions and S.rows==nvariables:
        print "S=okay"
    elif S.cols!=nreactions:
        print "Wrong number of reactions in S"
    elif S.rows!=nvariables:
        print "Wrong number of variables in S"

    return (S,a,nreactions,nvariables,ymat,Mumat, c)


def make_T_matrix(nvariables, nreactions, TE_matrix, S):
    #TODO AFAIK, the variable T is unused. If this is true, this function is unnecessary
    T = []
    for nv in range(0,nvariables):
        row = []
        for nr in range(0,nreactions):
            Stmp = S[nv,nr]
            Ttmp = TE_matrix[nr,:]
            row.append(Stmp*Ttmp)
        T.append(row)
    return T

def substitute_mean_with_y(mom, nvariables):

    """
    Replaces first order raw moments(e.g. x01, x10) by explicit means (e.g. y_0, y_1)

    :param mom: the list of expressions (moments)
    :param nvariables: the number of species
    :return: the substituted expressions
    """


    # e.g. x001,x010 x100 are means
    for i in range(0, nvariables):
        numv = [0] * nvariables
        numv[i] = 1
        numstr = str(numv[0])
        for j in range(1, nvariables):
            numstr = numstr + str(numv[j])
        t1 = Symbol('y_%d' %(i))
        t2 = Symbol('x'+numstr)

        for m in range(0, len(mom)):
            if isinstance(mom[m],list):
                for n in range(0, len(mom[m])):
                    mom[m][n] = Subs(mom[m][n], t2, t1).doit()
            else:
                mom[m] = Subs(mom[m], t2, t1).doit()

    return mom

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

    for i in range(len(momvec)-1,-1,-1):
        string = str(momvec[i])

        num = string[2:]

        # mm is the raw moment
        mm = Symbol('x'+num)
        soln = solve(mom[i] - momvec[i], mm)
        for m in range(0,len(CentralMoments)):
            for n in range(0,len(CentralMoments[m])):
                CentralMoments[m][n] = Subs(CentralMoments[m][n],mm, soln).doit()
    return CentralMoments

def substitute_ym_with_yx(CentralMoments, momvec):


    """
    Substitute central moment terms ymn, where n gives n1,...nd combination
    for yxi where i indicates index in counter for that n1,...,nd

    :param CentralMoments:
    :param momvec: the symbols for central moments ()
    :return: the symbols for central moments (e.g. ym11, ym02, ...)
    """


    for i in range(0,len(momvec)):
        yx = Symbol('yx'+str(i+1))

        for m in range(0, len(CentralMoments)):
            for n in range(0, len(CentralMoments[m])):
                CentralMoments[m][n] = Subs(CentralMoments[m][n], momvec[i], yx).doit()
                try:
                    CentralMoments[m][n] = simplify(CentralMoments[m][n])
                except:
                    pass
    return CentralMoments

def make_mfk(CentralMoments, yms, M):
    #TODO figure-out what MFK stands for

    """

    :param CentralMoments:
    :param yms:
    :param M:
    :return: MFK ...
    """

    # Get expressions for higher order central moments
    MFK1 = M*yms

    MFK = []
    # Reshape to a vector
    for i in range(0,len(MFK1)):
        try:
            MFK1[i] = simplify(MFK1[i])
        except:
            pass
        MFK.append(MFK1[i])


    for i in range(0,len(CentralMoments)):
        rowmat = Matrix(1, len(CentralMoments[i]), CentralMoments[i])

        # This should be a scalar
        MFK2 = rowmat * yms
        # TODO scalar => {len() == 1} so why do we need a loop here ?
        for j in range(0,len(MFK2)):
            try:
                MFK2[j] = simplify(MFK2[j])
            except:
                pass
            MFK.append(MFK2[j])
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


    # Define the kinetic model
    (S, amat, nreactions, nvariables, ymat, Mumat, c) = get_and_check_model(model_filename)

    # Make the derivation matrix
    ## damat = make_damat(amat, nMoments, ymat) TODO


    # compute counter and mcounter; the "k" and "n" vectors in equations. counter = mcounter - first_order_moments
    (counter, mcounter) = fcount(nMoments, nvariables)
    # Calculate TaylorExpansion terms to use in dmu/dt (eq. 6)
    TE_matrix = taylor_expansion(ymat, amat, counter)



    # M is the product of the stoichiometry matrix by the Taylor Expansion terms.
    # one row per species and one col per element of counter
    M = S*TE_matrix

    # T seems unused, TODO
    T = make_T_matrix(nvariables, nreactions, TE_matrix, S)


    #  Calculate expressions to use in central moments equations (eq. 9)
    #  CentralMoments is a list with entry for each moment (n1,...,nd) combination.
    # TODO remove `nDerivatives`: it is simply an alias for nMoments
    nDerivatives = nMoments
    CentralMoments = eq_centralmoments(counter,mcounter, M,T,nvariables,ymat,nreactions,nMoments,amat,S,nDerivatives)


    #  Substitute means in CentralMoments by y_i (ymat entry)
    CentralMoments = substitute_mean_with_y(CentralMoments, nvariables)


    #  Substitute higher order raw moments in terms of central moments
    #  raw_to_central calculates central moments (momvec) in terms
    #  of raw moment expressions (mom) (eq. 8)
    (mom, momvec) = raw_to_central(nvariables, counter, ymat, mcounter)



    # Substitute one for zeroth order raw moments in mom
    # TODO apparently unnecessary outer loop
    for i in range(0, nvariables):
        numv = [0] * nvariables
        numstr = str(numv[0])
        for j in range(1, nvariables):
            numstr = numstr + str(numv[j])

        t1 = F(1)
        t2 = Symbol('x'+numstr)
        for m in range(0, len(mom)):
            mom[m] = Subs(mom[m], t2, t1).doit()


    # Substitute first order raw moments (means) in mom with y_i (ymat entry)
    mom = substitute_mean_with_y(mom,nvariables)

    # Substitute raw moment, in CentralMoments, with of central moments
    CentralMoments = substitute_raw_with_central(CentralMoments, momvec, mom)

    # Use counter index (c) for yx (yxc) instead of moment (ymn) (e.g. ym021)
    CentralMoments = substitute_ym_with_yx(CentralMoments, momvec)

    # Make yms; (yx1, yx2, yx3,...,yxn) where n is the number of elements in counter
    if len(CentralMoments) != 0:
        nM = len(CentralMoments[0])
    else:
        nM = 1

    yms = Matrix(nM, 1, lambda i, j : var('yx%d' % i))
    # Set zeroth order central moment to 1
    yms[0] = 1

    # Get expressions for each central moment, and enter into list MFK
    MFK = make_mfk(CentralMoments, yms, M)


    # Write information to output file (and moment names and equations to .tex file)
    write_output(str(sys.argv[3]), nvariables, nMoments, counter, c, yms, ymat, MFK, time() - time1)


def get_args():
    model_ = sys.argv[1]
    numMoments = int(sys.argv[2])

    return (model_, numMoments)


if __name__ == "__main__":
    model_, numMoments = get_args()
    MFK_final(model_, numMoments)
