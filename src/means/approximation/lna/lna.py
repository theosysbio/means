"""
Linear Noise Approximation
-----

This part of the package implements Linear Noise Approximation as described in [Komorowski2009]_.

Example:

>>> from means.approximation.lna.lna import lna_approximation
>>> from means.examples.sample_models import MODEL_P53
>>> ode_problem = lna_approximation(MODEL_P53)
>>> print ode_problem

The result is an :class:`means.core.problems.ODEProblem`. Typically, it would be further used to
perform simulations (see :mod:`~means.simulation`) and inference (see :mod:`~means.inference`).

.. [Komorowski2009] M. Komorowski, B. Finkenstadt, C. V. Harper, and D. A. Rand,\
"Bayesian inference of biochemical kinetic parameters using the linear noise approximation,"\
BMC Bioinformatics, vol. 10, no. 1, p. 343, Oct. 2009.

------------
"""
import operator

import sympy as sp

from means.approximation.approximation_baseclass import ApproximationBaseClass
from means.core import Moment, VarianceTerm, ODEProblem


def lna_approximation(model):

    r"""
    A wrapper around :class:`~means.approximation.lna.lna.LinearNoiseApproximation`.
    It performs linear noise approximation (MEA).

    :return: an ODE problem which can be further used in inference and simulation.
    :rtype: :class:`~means.core.problems.ODEProblem`
    """
    lna = LinearNoiseApproximation(model)
    return lna.run()


class LinearNoiseApproximation(ApproximationBaseClass):
    """
    A class to performs Linear Noise Approximation of a model.
    """
    def run(self):
        """
        Overrides the default _run() private method.
        Performs the complete analysis
        :return: A fully computed set of Ordinary Differential Equations that can be used for further simulation
        :rtype: :class:`~means.core.problems.ODEProblem`
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
                A[i, k] = reduce(operator.add, [S[i, j] * sp.diff(amat[j], ymat[k]) for j in range(len(amat))])


        # `diagA` is a matrix that has values sqrt(a[i]) on the diagonal (0 elsewhere)
        diagA = sp.Matrix(len(amat), len(amat), lambda i, j: amat[i] ** sp.Rational(1,2) if i==j else 0)
        # E is stoichiometry matrix times diagA
        E = S * diagA

        variance_terms = []
        cov_matrix = []
        for i in range(len(ymat)):
            row = []
            for j in range(len(ymat)):
                if i <= j:
                    symbol = 'V_{0}_{1}'.format(i, j)
                    variance_terms.append(VarianceTerm(position=(i,j), symbol=symbol))
                else:
                    # Since Vi,j = Vj,i, i.e. covariance are equal, we remove the repetitive terms
                    symbol = 'V_{0}_{1}'.format(j, i)
                    variance_terms.append(VarianceTerm(position=(j,i), symbol=symbol))
                row.append(symbol)
            cov_matrix.append(row)

        V = sp.Matrix(cov_matrix)


        # Matrix of variances (diagonal) and covariances of species i and j differentiated wrt time.
        # I.e. if i=j, V_ij is the variance, and if i!=j, V_ij is the covariance between species i and species j
        dVdt = A * V + V * (A.T) + E * (E.T)


        # build ODEProblem object
        rhs_redundant = sp.Matrix([i for i in dPdt] + [i for i in dVdt])


        #generate ODE terms
        n_vectors = [tuple([1 if i==j else 0 for i in range(n_species)]) for j in range(n_species)]
        moment_terms = [Moment(nvec,lhs) for (lhs, nvec) in zip(ymat, n_vectors)]

        ode_description = moment_terms + variance_terms


        non_redundant_idx = []
        ode_terms = []
        # remove repetitive covariances, as Vij = Vji
        for i, cov in enumerate(ode_description):
            if cov in ode_terms:
                continue
            else:
                ode_terms.append(cov)
                non_redundant_idx.append(i)
        rhs = []
        for i in non_redundant_idx:
            rhs.append(rhs_redundant[i])


        out_problem = ODEProblem("LNA", ode_terms, rhs, sp.Matrix(self.model.parameters))


        return out_problem
