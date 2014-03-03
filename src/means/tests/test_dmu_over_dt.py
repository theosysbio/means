import unittest
import sympy as sp
from means.approximation.mea.dmu_over_dt import generate_dmu_over_dt
from means.approximation.ode_problem import Moment
from means.approximation.mea.moment_expansion_approximation import MomentExpansionApproximation
from means.util.sympyhelpers import to_sympy_matrix, assert_sympy_expressions_equal


class TaylorExpansionTestCase(unittest.TestCase):

    def test_TaylorExpansion(self):
        """
        Given the number of moments is 3, the number of species is 2,
        Given the propensities of the 3 reactions in `a_strings`,
        And Given the combination of derivative order in counter,
        Then results of `TaylorExpansion()` should produce a matrix exactly equal to
        exactly equal to the the expected one (`expected_te_matrix`).

        :return:
        """

        mea = MomentExpansionApproximation(None, 3)
        species = ["a", "b", "c"]
        propensities = to_sympy_matrix(["a*2 +w * b**3", "b - a*x /c", "c + a*b /32"])
        stoichiometry_matrix = sp.Matrix([
            [1, 0, 1],
            [-1, -1, 0],
            [0, 1, -1]
        ])

        counter = [
            Moment([0, 0, 2], sp.Symbol("q1")),
            Moment([0, 2, 0], sp.Symbol("q2")),
            Moment([0, 0, 2], sp.Symbol("q3")),
            Moment([2, 0, 0], sp.Symbol("q4")),
            Moment([1, 1, 0], sp.Symbol("q5")),
            Moment([0, 1, 1], sp.Symbol("q6")),
            Moment([1, 0, 1], sp.Symbol("q7"))]

        result = generate_dmu_over_dt(species, propensities, counter, stoichiometry_matrix)
        expected = stoichiometry_matrix * to_sympy_matrix([["        0", "3*b*w", "0", "0", "0", "0", "0"],
                                                           ["-a*x/c**3", "0", "-a*x/c**3", "0", "0", "0", "x/c**2"],
                                                           ["0", "0", "0", "0", "1/32", "0", "0"]])

        self.assertEqual(result, expected)
