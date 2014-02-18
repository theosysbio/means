import unittest
import random
import numpy as np
from numpy.testing import assert_array_almost_equal
from means.inference.hypercube import hypercube


class TestHyperCube(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        np.random.seed(42)


    def test_hypercube_regression_dimer(self):

        param_ranges = [(0.001, 0.0), (0.5, 0.0), (330.0, 260.0)]
        initial_conditions_ranges = [(320.0, 290.0)]

        ans = hypercube(5, param_ranges + initial_conditions_ranges)
        expected_ans = [[0.0003553578523702354, 0.29734640303161364, 306.2260484701648, 304.7826314512718],
                        [5.270575716719754e-05, 0.4781362025196397, 318.7185304743407, 296.16130541612375],
                        [0.0008721146403084233, 0.34946447118966373, 285.8232870026351, 309.6216092798371],
                        [0.0005449941363261761, 0.03501155622204766, 260.59901698910505, 293.7287937367499],
                        [0.0007949978489554667, 0.1801162349313351, 297.2364927687481, 315.1572303603537]]

        assert_array_almost_equal(ans, expected_ans, decimal=12)