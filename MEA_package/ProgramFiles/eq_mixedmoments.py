#####################################################################
# Called by centralmoments.py
# Provides the terms needed for equation 11 in Angelique's paper
# This gives the expressions for dB/dt in equation 9, these are the 
# time dependencies of the mixed moments
####################################################################

from sympy import Symbol, Matrix, diff
from sympy import S as F
from math import factorial

def eq_mixedmoments(nreactions,nvariables,nMoments,amat,counter,S,ymat,nDerivatives,kvec,ekcounter,dAdt):
    
    mixedmomentst=Matrix(len(ekcounter),dAdt.cols,lambda i,j:0)
    for reaction in range(0,nreactions):

        mixedmoments=Matrix(len(ekcounter),dAdt.cols,lambda i,j:0)
        for ekloop in range(0,len(ekcounter)):

            evec = ekcounter[ekloop]
            Enumber = sum(evec)
            if Enumber>0:
                ##########################################
                # s^ e terms in eq. 11
                ##########################################
                f_1=1          
                for fi in range(0,len(evec)):

                    f_1 = f_1*S[fi,reaction]**evec[fi]

                ##########################################
                # (k e) binomial terms in eq 11
                #########################################
                f_2 = 1        
                for fi in range(0,len(evec)):
                    f_2=f_2*factorial(kvec[fi])/(factorial(evec[fi])*factorial(kvec[fi]-evec[fi]))

                ########################################
                # x^(k-e) terms in eq 11
                ########################################
                yterms=(ymat[0])**(kvec[0]-evec[0])

                for nv in range(1,nvariables):
                    yterms=yterms*(ymat[nv])**(kvec[nv]-evec[nv])
                E=yterms*amat[reaction]   #expectation value terms

                
                ########################################
                # Derivatives of dF/dt in equation 12 in 
                # Angelique's paper
                #######################################                

                dEmat = Matrix(nDerivatives,1,lambda i,j:0)
                for D in range(0,nDerivatives):
                    if D==0:
                        row = []
                        for nv in range(0,nvariables):
                            deriv = diff(E,ymat[nv])
                            row.append(deriv)
                        dEmat[D,0] = row
                    else:
                        prev = Matrix(dEmat[D-1,0])
                        row = []
                        y = len(prev)
                        for eq in range(0,y):
                            for nv in range(0,nvariables):
                                deriv = diff(prev[eq],ymat[nv])
                                row.append(deriv)
                        dEmat[D,0] = row
                

                #########################################
                # Loops over necessary combinations of
                # moments to calculate terms for equation
                # 12 in paper
                #########################################
                
                TE = Matrix(len(counter),1,lambda i,j:0)
                for Te in range(0,len(counter)):

                    nvec = counter[Te]
                    Dnumber = sum(nvec)

                    if Dnumber==0:
                        TE[Te] = E
                    if Dnumber>1 and Dnumber<(nMoments+1):
                        r_1 = 1
                        for j in nvec:
                            r_1 = r_1*factorial(j)
                        n=nvec
                        nzidx = []
                        for i in range(0,len(n)):
                            if n[i]!=0:
                                nzidx.append(i)
                        nnew = []
                        for i in range(0,len(nzidx)):
                            idx = nzidx[i]
                            for j in range(0,n[idx]):
                                nnew.append(idx)
                        Didx=0
                        for nzs in range(0,len(nnew)):

                            Didx = Didx + ((nvariables)**(Dnumber-nzs-1))*(nnew[nzs])

                        TE[Te] = F(1)/r_1*dEmat[Dnumber-1][Didx]

                    Taylorexp=f_1*f_2*TE

                for i in range(0,mixedmoments.cols):
                    mixedmoments[ekloop,i]=Taylorexp[i]
        #####################################################
        # mixedmomentst is a matrix with a row for each e1,..,ed
        # combindation in eq.11.  Columns are summed over to
        # give mixedmoments which gives dB/dt terms for use in 
        # eq. 9.  mixedmoments is a vector with an entry for each
        # k1,...,kd combination in eq.9
        ######################################################

        mixedmomentst=mixedmomentst+mixedmoments
    mixedmoments=Matrix(1,mixedmomentst.cols,lambda i,j:sum(mixedmomentst[:,j]))
    return mixedmoments
