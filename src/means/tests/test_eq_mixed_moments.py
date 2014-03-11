import unittest

import sympy

from means.approximation.mea.eq_mixed_moments import DBetaOverDtCalculator
from means.core import Moment
from means.util.sympyhelpers import to_sympy_matrix, assert_sympy_expressions_equal


class TestEqMixedMoments(unittest.TestCase):

    def test_for_p53(self):
        """
        Given the preopensities,
        Given the soichiometry matrix,
        Given the counter (list of Moments),
        Given the species list,
        Given k_vector and
        Given ek_counter (list of moment)
        The answer should match exactly the expected result
        :return:
        """

        stoichio = sympy.Matrix([
            [1, -1, -1, 0,  0,  0],
            [0,  0,  0, 1, -1,  0],
            [0,  0,  0, 0,  1, -1]
        ])

        propensities = to_sympy_matrix([
            ["                    c_0"],
            ["                c_1*y_0"],
            ["c_2*y_0*y_2/(c_6 + y_0)"],
            ["                c_3*y_0"],
            ["                c_4*y_1"],
            ["                c_5*y_2"]])

        counter = [
        Moment([0, 0, 0], 1),
        Moment([0, 0, 2], sympy.Symbol("yx1")),
        Moment([0, 1, 1], sympy.Symbol("yx2")),
        Moment([0, 2, 0], sympy.Symbol("yx3")),
        Moment([1, 0, 1], sympy.Symbol("yx4")),
        Moment([1, 1, 0], sympy.Symbol("yx5")),
        Moment([2, 0, 0], sympy.Symbol("yx6"))
        ]

        species = sympy.Matrix(["y_0", "y_1", "y_2"])

        dbdt_calc = DBetaOverDtCalculator(propensities, counter,stoichio, species)
        k_vec = Moment([1, 0, 0], None)
        ek_counter = [Moment([1, 0, 0], sympy.Symbol("y_0"))]

        answer = dbdt_calc.get(k_vec,ek_counter).T
        result = to_sympy_matrix(["c_0 - c_1*y_0 - c_2*y_0*y_2/(c_6 + y_0)"," 0"," 0"," 0"," c_2*y_0/(c_6 + y_0)**2 - c_2/(c_6 + y_0)"," 0"," -c_2*y_0*y_2/(c_6 + y_0)**3 + c_2*y_2/(c_6 + y_0)**2"])
        assert_sympy_expressions_equal(answer, result)

