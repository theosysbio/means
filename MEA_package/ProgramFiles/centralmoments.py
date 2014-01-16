##################################################################
# Used to calculate terms required for use in the equations giving
# time dependence of central moments (Equation 9 in Angelique's 
# paper).  
#
# Returns list 'CentralMoments' with an entry for each moment
# (n1,...,nd) combination
##################################################################

from sympy import Matrix, Symbol
from math import factorial
from eq_mixedmoments import eq_mixedmoments

def eq_centralmoments(counter,mcounter,M,TaylorM,nvariables,ymat,nreactions,nMoments,amat,S,nDerivatives):
    
    mixmom = 0
    ncounter = counter[:]
    ncounter.remove(counter[0])    #counter w/o zeroth order moment
    centralmoments = []

    ###############################################################
    # Loops through required combinations of moments (n1,...,nd)
    # (does not include 0th order central moment as this is 1,
    # or 1st order central moment as this is 0
    ###############################################################

    for Tn in range(0,len(ncounter)):
        centralmomentsTn = []
        nvec = ncounter[Tn]   #the moment
        repmat = []
        for i in range(0,len(mcounter)):
            repmat.append(nvec)
        G = Matrix(mcounter)
        H = Matrix(repmat)
        check = G - H
        midx = []
        for i in range(0,check.rows):
            mc = max(check[i,:])
            if mc<=0:
                midx.append(i)
        Taylorexp = [[0]*len(counter)]*len(mcounter)
        for Tm in range(0,len(midx)):
            mvec = mcounter[midx[Tm]]

            ##########################################
            # f_2 is the (n k) binomial term in equation 9
            #########################################
            f_2=1
            for fi in range(0,len(mvec)):
                f_2 = f_2*factorial(nvec[fi])/(factorial(mvec[fi])*factorial(nvec[fi]-mvec[fi]))
            ##########################################
            # f_3 is (-1)^(n-k) term in equation 9
            ##########################################
            f_3 = 1
            for fi in range(0,len(mvec)):
                f_3=f_3*(-1)**(nvec[fi]-mvec[fi])
            

            ##########################################
            # Calculate A, dAdt terms in equation 9
            # (equivalent to fA_counter in Angelique's code)
            # ymat used in place of means - these will be replaced later
            ##########################################
            A = (ymat[0])**(nvec[0]-mvec[0])
            for nv in range(1, nvariables):
                A = A*(ymat[nv]**(nvec[nv]-mvec[nv]))
            
            dAdt = (nvec[0]-mvec[0])*(ymat[0]**(-1))*A*M[0,:]
            for nv in range(1, nvariables):
                dAdt = dAdt + (nvec[nv]-mvec[nv])*(ymat[nv]**(-1))*A*M[nv,:]

            
            ##########################################
            # Calculate B, dBdt terms in equation 9
            ##########################################

            mstr = str(mvec[0])
            for mm in range(1,len(mvec)):
                mstr = mstr+str(mvec[mm])
            B = Symbol('x'+mstr)
            repmat = []
            for i in range(0,len(mcounter)):
                repmat.append(mvec)

            C = Matrix(mcounter)
            D = Matrix(repmat)
            check = C - D

            ekidx = []
            for i in range(0,check.rows):
                mc = max(check[i,:])
                if mc<1:
                    ekidx.append(i)

            ekcounter = []
            for i in range(0,len(ekidx)):
                ekcounter.append(mcounter[ekidx[i]])

            for i in range(0,len(ekcounter)):
                if sum(ekcounter[i])==0:
                    ekcounter[i] = "zero"

            ekcounter.remove("zero")
            if ekcounter != []:
                dBdt = eq_mixedmoments(nreactions,nvariables,nMoments,amat,counter,S,ymat,nDerivatives,mvec,ekcounter,dAdt)

            else:

                dBdt = Matrix(dAdt.rows,dAdt.cols,lambda i,j:0)
                B=1

     
            mixmom = mixmom+1

            Taylorexp[Tm] = (f_2*f_3*(A*dBdt+B*dAdt))

        #################################################
        # Create list of central moments terms to return
        # to MEA program
        # Taylorexp is a matrix which has an entry equal to
        # the f2*f3*(AdB/dt + B dA/dt) term in equation 9
        # for each k1,..,kd
        # These are summed over to give the Taylor Expansion
        # for each n1,..,nd combination in equation 9
        #################################################
        Taylorexp1 = Matrix(Taylorexp)
        for j in range(0,len(counter)):
            centralmomentsTn.append(sum(Taylorexp1[:,j]))
        centralmoments.append(centralmomentsTn)
        
    return centralmoments
