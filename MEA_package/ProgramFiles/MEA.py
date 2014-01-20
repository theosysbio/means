####python MFK_final.py <model> <nMoments> <outputfile>

from time import time
import sys
import os

from sympy import Matrix, diff, Symbol, Subs, Eq, var, simplify
model_ = sys.argv[1]
os.system('python formatmodel.py '+model_)

#############################################################################
#   MFK_final takes number of moments as input, and produces central
#   moment equations using the specified model
#############################################################################



def make_damat(amat, nMoments, nreactions, nvariables,ymat):

    ##############################################################
    # Calculate matrix of derivatives of rate equations ("damat")
    # (n-1)th row gives nth order derivatives
    # number of columns = nreactions
    # each entry is a list with derivatives for that reaction/order
    ##############################################################


    # In the end, damat contains the derivatives of all orders (ord), for all reactions (react),
    # with respect to all species (sps).
    # damat[ord][react][sps]


    nDerivatives = nMoments
    damat = Matrix(nDerivatives,1, lambda i,j:0)


    # At this point, `damat` is a matrix with 1 col and as many rows as moments. It is filled with 0s
    # `amat` is the column matrix of propensities (as many as reactions)
    # `ymat` is the column matrix of species (variables)


    # For all moment orders
    for D in range(0,nDerivatives):
        # if 0th order moment
        if D==0:
            # create an empty row
            row = []
            # For all reactions
            for na in range(0,nreactions):
                # create an empty vect of reactions
                reaction = []
                # for all variables/ all species
                for nv in range(0,nvariables):
                    # We derive the propensity of this reaction with respect to a species

                    deriv = diff(amat[na,0],ymat[nv,0])
                    reaction.append(deriv)
                # In the end, we get the partial derivatives of the propensity
                # of this reaction with respect to all species.
                row.append(reaction)
            # For all reactions in a given order of derivation D we have a row
            damat[D,0] = row
        else:
            # this does the same as above but does higher order derivatives from the results obtained before
            prev = Matrix(damat[D-1,0])
            row = []
            for na in range(0,nreactions):
                reaction = []
                prevna = prev[na,:]
                y = len(prevna)
                for eq in range(0,y):
                    for nv in range(0,nvariables):
                        deriv = diff(prevna[0,eq],ymat[nv,0])
                        reaction.append(deriv)
                row.append(reaction)
            damat[D,0] = row
    return damat

def MFK_final(nMoments):
    time1 = time()
    output = open(str(sys.argv[3]),'w')
    from model import model
    from fcount import fcount
    from sympy import Matrix, diff, Symbol, Subs, Eq, var, simplify
    from sympy import S as F
    from sympy.solvers import solve
    from TaylorExpansion import TaylorExpansion
    from centralmoments import eq_centralmoments
    from raw_to_central import raw_to_central
    from sympy import latex



    #Define the kinetic model
    [S,a,nreactions,nvariables,ymat,Mumat, c] = model()

    #Delete temporary model file
    os.system('rm model.py*')

    #Check stoichiometry matrix
    if S.cols==nreactions and S.rows==nvariables:            
        print "S=okay"
    elif S.cols!=nreactions:
        print "Wrong number of reactions in S"
    elif S.rows!=nvariables:
        print "Wrong number of variables in S"



    amat = a
    damat = make_damat(a, nMoments, nreactions, nvariables, ymat)

    #####################################################################
    #  Calculate TaylorExpansion terms to use in dmu/dt (eq. 6)
    #####################################################################

    [counter, mcounter] = fcount(nMoments, nvariables)
    TE_matrix = TaylorExpansion(nreactions,nvariables,damat,amat,counter,nMoments)
    
    M = (S*TE_matrix) 
    
    T = []
    for nv in range(0,nvariables):
        row = []
        for nr in range(0,nreactions):
            Stmp = S[nv,nr]
            Ttmp = TE_matrix[nr,:]
            row.append(Stmp*Ttmp)
        T.append(row)
   

    #####################################################################
    #  Calculate expressions to use in central moments equations (eq. 9)
    #  CentralMoments is a list with entry for each moment (n1,...,nd) 
    #  combination.
    #####################################################################
    nDerivatives = numMoments
    CentralMoments = eq_centralmoments(counter,mcounter,M,T,nvariables,ymat,nreactions,nMoments,amat,S,nDerivatives)

    
    #####################################################################
    #  Substitute means in CentralMoments by y_i (ymat entry)
    ####################################################################

    for i in range(0, nvariables):
        numv = [0] * nvariables
        numv[i] = 1
        numstr = str(numv[0])
        for j in range(1, nvariables):
            numstr = numstr + str(numv[j])
        t1 = Symbol('y_%d' %(i))
        t2 = Symbol('x'+numstr)
        
        for m in range(0, len(CentralMoments)):
            for n in range(0, len(CentralMoments[m])):
                CentralMoments[m][n] = Subs(CentralMoments[m][n], t2, t1).doit()
    
    #####################################################################
    #  Substitute higher order raw moments in terms of central moments
    #  raw_to_central calculates central moments (momvec) in terms
    #  of raw moment expressions (mom) (eq. 8)
    #####################################################################   
    
    [mom, momvec] = raw_to_central(nvariables, counter, ymat, mcounter)
    
    # Substitute one for zeroth order raw moments in mom
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
    for i in range(0, nvariables):
        numv = [0] * nvariables
        numv[i] = 1
        numstr = str(numv[0])
        for j in range(1, nvariables):
            numstr = numstr + str(numv[j])
        t1 = Symbol('y_%d' %(i))
        t2 = Symbol('x'+numstr)
        for m in range(0, len(mom)):
            mom[m] = Subs(mom[m], t2, t1).doit()
            
    # Substitute raw moment terms in CentralMoments in terms of central moments
    # (need to iterate in reverse from highest to lowest order moments to ensure all
    # raw moments are replaced as some higher order raw moments are expressed in terms 
    # of central and lower order raw moments)

    for i in range(len(momvec)-1,-1,-1):
        string = str(momvec[i])
        num = string[2:]
        mm = Symbol('x'+num)
        soln = solve(mom[i]-momvec[i],mm)

        for m in range(0,len(CentralMoments)):
            for n in range(0,len(CentralMoments[m])):
                CentralMoments[m][n] = Subs(CentralMoments[m][n],mm, soln).doit()
    
    ##############################################################################
    # Substitute central moment terms ymn, where n gives n1,...nd combination
    # for yxi where i indicates index in counter for that n1,...,nd
    ##############################################################################

    if len(CentralMoments) != 0:
        nM = len(CentralMoments[0])
    else:
        nM = 1
    yms = Matrix(nM,1,lambda i,j:var('yx%d' % i))

    for i in range(0,len(momvec)):
        yx=Symbol('yx'+str(i+1))
        for m in range(0, len(CentralMoments)):
            for n in range(0, len(CentralMoments[m])):
                CentralMoments[m][n] = Subs(CentralMoments[m][n], momvec[i], yx).doit()
                try:
                    CentralMoments[m][n] = simplify(CentralMoments[m][n])
                except:
                    pass
    
    ##############################################################################
    # Get expressions for each central moment, and enter into list MFK
    ##############################################################################
    
    # Set zeroth order central moment to 1
    yms[0] = 1
 
    # Get expressions for higher order central moments
    MFK1 = M*yms
    MFK = []
    for i in range(0,len(MFK1)):
        try:
            MFK1[i] = simplify(MFK1[i])
        except:
            pass
        MFK.append(MFK1[i])
        
    for i in range(0,len(CentralMoments)):
        rowmat = Matrix(1,len(CentralMoments[i]),CentralMoments[i])
        MFK2 = rowmat*yms
        for j in range(0,len(MFK2)):
            try:
                MFK2[j] = simplify(MFK2[j])
            except:
                pass
            MFK.append(MFK2[j])


    ###############################################################################
    # Write information to output file (and moment names and equations to .tex file)
    ################################################################################

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
 
    out_tex = open(str(sys.argv[3])+'.tex','w')
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
    time2 = time()
    time3 = time2-time1
    output.write('Number of variables:\n'+str(nvariables)+'\nNumber of moments:\n'+str(nMoments)+'\nTime taken (s): '+str(time3))
    
    output.write('\n\nNumber of equations:\n'+str(len(LHS)))
    output.write('\n\nList of moments:\n')
    for i in range(len(moment_list)):
        output.write(str(moment_list[i])+'\n')
        
    output.close()
    
    out_tex.write('\n\end{document}')
    out_tex.close()

numMoments = int(sys.argv[2])
MFK_final(numMoments)
