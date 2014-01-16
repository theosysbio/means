#######################################################################
#  Expresses higher (2+) order raw moments in terms of central moments
#  Returns momvec (list of central moments, ymi) and mom (list of
#  equivalent expressions in terms of raw moments) (see eq. 8)
#######################################################################

from sympy import Matrix, Symbol
from math import factorial

def raw_to_central(nvariables, counter, ymat, mcounter):
    
    ncounter = counter[:]
    ncounter.remove(counter[0])

    mom = []        #create empty list for mom

    for Tn in range(0,len(ncounter)):  #loop through all n1,...,nd combinations
        nvec = ncounter[Tn]            
        repmat = []
        for i in range(0,len(mcounter)):
            repmat.append(nvec)
        G = Matrix(mcounter)
        H = Matrix(repmat)
        check = G - H

        midx = []                      #midx is list of mvecs <= current nvec
        for i in range(0,check.rows):
            mc = max(check[i,:])
            if mc<=0:
                midx.append(i)

        ################################################################
        #  Calculate terms in eq. 8 to sum over for each n1,...,nd combination
        #  Each entry in Taylorexp corresponds to a k1,...,kd combination
        ################################################################

        Taylorexp = Matrix(len(midx),1, lambda i,j:0)
        
        for Tm in range (0, len(midx)):
            mvec = mcounter[midx[Tm]]
            
            f_2=1                                       #f_2 is (n k) binomial term
            for fi in range(0,len(mvec)):
                f_2 = f_2*factorial(nvec[fi])/(factorial(mvec[fi])*factorial(nvec[fi]-mvec[fi]))
            
            f_3 = 1                                      #f_3 is (-1)^(n-k) term
            for fi in range(0,len(mvec)):
                f_3=f_3*(-1)**(nvec[fi]-mvec[fi])

            A = (ymat[0])**(nvec[0]-mvec[0])             #alpha term: mu^(n-k)
            for nv in range (1, nvariables):
                A = A* (ymat[nv]**(nvec[nv]-mvec[nv]))
                
            mstr = str(mvec[0])                          #beta term: x^k
            for mm in range(1, len(mvec)):
                mstr = mstr + str(mvec[mm])
            B = Symbol('x'+mstr)

            Taylorexp[Tm] = f_2 *f_3 *(A*B)              #calculate term for k1,....,kd

        mom.append(sum(Taylorexp))     #sum over k1,...,kd terms for particular n1,...,nd
        
    momvec = []
    for i in range (0, len(ncounter)):
        vec = ncounter[i]
        mstr = str(vec[0])
        for mm in range(1, len(vec)):
            mstr = mstr + str(vec[mm])          #creates string of indices for each n1,...,nd
        momvec.append(Symbol('ym'+mstr))
    
    return(mom, momvec)        
