from sympy import Matrix, Symbol, diff, latex

def LNA(S, a, ymat):

    # dPdt is matrix of each species differentiated w.r.t. time
    # The code below literally multiplies the stoichiometry matrix to a column vector of propensities
    # from the right (::math::`\frac{dP}{dt} = \mathbf{Sa}`)
    dPdt = Matrix(len(ymat), 1, lambda i, j: 0)
    for i in range(len(ymat)):
        dPidt = S[i, 0] * a[0]
        for j in range(1, len(a)):
            dPidt += S[i, j] * a[j]
        dPdt[i] = dPidt

    # A Is a matrix of each species (rows) and the derivatives of their stoichiometry matrix rows
    # against each other species
    # Code below computes the matrix A, that is of size `len(ymat) x len(ymat)`, for which each entry
    # ::math::`A_{ik} = \sum_j S_{ij} \frac{\partial a_j}{\partial y_k} = \mathfb{S_i} \frac{\partial \mathbf{a}}{\partial y_k}`
    A = Matrix(len(ymat), len(ymat), lambda i, j: 0)
    for i in range(A.rows):
        for k in range(A.cols):
            Aik = S[i, 0] * diff(a[0], ymat[k])
            for j in range(1, len(a)):
                Aik += S[i, j] * diff(a[j], ymat[k])
            A[i, k] = Aik

        #    E = Matrix(len(ymat),len(a),lambda i,j:0)
        #    for i in range(len(ymat)):
        #        for k in range(len(a)):
        #            E[i,k] = S[i,k]*(a[k]**(-0.5))

    # `diagA` is a matrix that has values sqrt(a[i]) on the diagonal
    diagA = Matrix(len(a), len(a), lambda i, j: 0)
    for i in range(len(a)):
        for j in range(len(a)):
            if i == j:
                diagA[i, j] = a[i] ** 0.5

    # E is stoichiometry matrix times diagA
    E = S * diagA

    # V is a matrix of symbols V_ij for all i and j (TODO: this won't work for more than 10 species)
    V = Matrix(len(ymat), len(ymat), lambda i, j: 'V_' + str(i) + str(j))  # TODO: (from original authors) Make V_ij equal to V_ji

    # Matrix of variances (diagonal) and covariances of species i and j differentiated wrt time.
    # I.e. if i=j, V_ij is the variance, and if i!=j, V_ij is the covariance between species i and species j
    dVdt = A * V + V * (A.T) + E * (E.T)

    # Generate moments list
    # This just returns all possible vectors with only first-order moments
    # (e.g. [1,0,0], [0,1,0], [0,0,1] in three species case)
    momlist = [0] * len(ymat)
    for i in range(len(ymat)):
        momlist_i = [0] * len(ymat)
        momlist_i[i] = 1
        momlist[i] = momlist_i

    return dPdt, dVdt, V, momlist

def print_output(LNAout, dPdt, dVdt, ymat, V, c, momlist):
    output = open(LNAout, 'w')
    output.write('LNA\n\nRHS of equations:\n')
    for i in dPdt:
        output.write(str(i) + '\n')
    for i in dVdt:
        output.write(str(i) + '\n')
    output.write('\nLHS:\n')
    for i in ymat:
        output.write(str(i) + '\n')
    for i in V:
        output.write(str(i) + '\n')
    output.write('\nConstants:\n')
    for i in c:
        output.write(str(i) + '\n')
    output.write('Number of variables:\n' + str(len(ymat)))
    output.write('\n\nNumber of equations:\n' + str(len(dPdt) + len(dVdt)) + '\n')
    output.write('\nList of moments:')
    for i in range(len(momlist)):
        output.write('\n' + str(momlist[i]))
    output.close()

    out_tex = open(LNAout + '.tex', 'w')
    out_tex.write(
        '\documentclass{article}\n\usepackage[landscape, margin=0.5in, a3paper]{geometry}\n\\begin{document}\n\section*{RHS of equations}\n')
    for i in range(len(dPdt)):
        out_tex.write('$\dot ' + str(latex(ymat[i])) + ' = ' + str(latex(dPdt[i])) + '$\\\\')
    for i in range(len(dVdt)):
        out_tex.write('$\dot ' + str(latex(V[i])) + ' = ' + str(latex(dVdt[i])) + '$\\\\')
    out_tex.write('\n\section*{Moments}\n')
    for i in range(len(momlist)):
        out_tex.write('\n$' + str(latex(ymat[i])) + '$: {' + str(momlist[i]) + '}\\\\')
    out_tex.write('\n\n\end{document}')
    out_tex.close()

if __name__ == '__main__':

    import sys

    model_ = sys.argv[1]
    LNAout = sys.argv[2]

    from model import parse_model

    model = parse_model(model_)

    dPdt, dVdt, V, momlist = LNA(model.stoichiometry_matrix, model.propensities, model.species)
    print_output(LNAout, dPdt, dVdt, model.species, V, model.constants, momlist)

