import operator

import sympy as sp

from means.approximation import ode_problem
from means.approximation.approximation_baseclass import ApproximationBaseClass
from means.approximation.ode_problem import Moment, VarianceTerm


class LinearNoiseApproximation(ApproximationBaseClass):
    """
    Performs Linear Noise Approximation of a model. todo add ref here
    """
    def _wrapped_run(self):
        """
        Overrides the default _run() private method.
        Performs the complete analysis
        :return: an ODEProblem which can be further used in inference and simulation
        """

        S = self.model.stoichiometry_matrix
        amat = self.model.propensities
        ymat = self.model.species
        n_species = len(ymat)

        # dPdt is matrix of each species differentiated w.r.t. time
        # The code below literally multiplies the stoichiometry matrix to a column vector of propensities
        # from the right (::math::`\frac{dP}{dt} = \mathbf{Sa}`)
        dPdt =  S * amat


        # A Is a matrix of each species (rows) and the derivatives of their stoichiometry matrix rows
        # against each other species
        # Code below computes the matrix A, that is of size `len(ymat) x len(ymat)`, for which each entry
        # ::math::`A_{ik} = \sum_j S_{ij} \frac{\partial a_j}{\partial y_k} = \mathfb{S_i} \frac{\partial \mathbf{a}}{\partial y_k}`
        A = sp.Matrix(len(ymat), len(ymat), lambda i, j: 0)
        for i in range(A.rows):
            for k in range(A.cols):
                A[i, k] = reduce(operator.add, [S[i, j] * sp.diff(amat[j], ymat[k])  for j in range(len(amat))])


        # `diagA` is a matrix that has values sqrt(a[i]) on the diagonal (0 elsewhere)
        diagA = sp.Matrix(len(amat), len(amat), lambda i, j: amat[i] ** 0.5 if i==j else 0)

        # E is stoichiometry matrix times diagA
        E = S * diagA

        # V is a matrix of symbols V_ij for all i and j (TODO: this won't work for more than 10 species)
        V = sp.Matrix(len(ymat), len(ymat), lambda i, j: 'V_' + str(i) + str(j))  # TODO: (from original authors) Make V_ij equal to V_ji

        # Matrix of variances (diagonal) and covariances of species i and j differentiated wrt time.
        # I.e. if i=j, V_ij is the variance, and if i!=j, V_ij is the covariance between species i and species j
        dVdt = A * V + V * (A.T) + E * (E.T)


        # build ODEProblem object
        rhs = sp.Matrix([i for i in dPdt] + [i for i in dVdt])

        #generate ODE terms
        n_vectors = [tuple([1 if i==j else 0 for i in range(n_species)]) for j in range(n_species)]
        moment_terms = [Moment(nvec,lhs) for (lhs, nvec) in zip(ymat, n_vectors)]

        variance_terms = [VarianceTerm(var) for var in V]
        ode_terms = moment_terms + variance_terms


        out_problem = ode_problem.ODEProblem("LNA", ode_terms, rhs, sp.Matrix(self.model.constants))
        return out_problem
