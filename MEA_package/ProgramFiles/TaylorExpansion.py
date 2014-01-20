########################################################################
#  TaylorExpansion function creates terms used in eq. 6
#  to calculate dmu/dt for each species/variable
#  (equivalent to taylorexp and iterate_counter functions in Matlab code
#  Returns TE_matrix: row for each reaction, column for n1,...,nd
#  combination in counter (i.e. all but 1st order)
########################################################################

from sympy import Matrix, Symbol, factorial
from sympy import S as F

def TaylorExpansion(nreactions, nvariables, damat, a, counter, nMoments):
    
    TE_matrix = Matrix(nreactions, len(counter), lambda i,j: 0)

    for L in range(0, nreactions):              #loop through each reaction 
        for i in range(0, len(counter)):        #loop through each entry in counter (n1,...,nd combination)
            nvec = counter[i]
            #Dnumber = (mixed) moment order
            Dnumber = sum(nvec)

            if Dnumber==0:                      #if no differentiation, add original propensity term
                TE_matrix[L,i] = a[L]
                
            elif (Dnumber>0 and Dnumber<=nMoments):  #if differentiation needed, calculate required term

                r_1 = 1                        #calculate factorial term (r_1 = n!)
                for j in nvec:
                    r_1 = r_1*factorial(j)

                #Calculate derivative term
                nz_idx = [k for k, n in enumerate(nvec) if n != 0]    #get indices for non-zero elements in nvec (n1,...,nd)


                nnew = []
                # repeat the index of non zero element in `nvec` as many time as their value and store it in `nnew`

                for k in range(0, len(nz_idx)):
                    idx = nz_idx[k]                         #idx = index in nvec
                    nvec_entry = nvec[idx]                  #nvec_entry = no. in nvec at index i (for xi ^ j, nvec_entry = j)
                    for j in range(0, nvec_entry):
                        nnew.append(idx)

                #Didx is index of required term in damat (differentiation) matrix
                Didx = 0
                for nzs in range(0, len(nnew)):
                    Didx = Didx + ((nvariables) ** (Dnumber - nzs - 1)) * (nnew[nzs])

                TE_matrix[L,i] = (F(1)/r_1)*damat[Dnumber-1][L][Didx]
    return (TE_matrix) 
