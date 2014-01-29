from fcount import fcount
import unittest
import itertools
class FcountTestCase(unittest.TestCase):
    def test_fcount_2_2(self):
        """
        Given the number of moments is 2, the number of species is 2,
        Then results of fcount.py should be 1) [[0,0],[0,1],[1,0],[2,0],[0,2],[1,1]], as the numbers sum to 2 or less
        2) [[0,0],[2,0],[0,2],[1,1]] if delete the first orders
        """
        fcount_result_without_firstorder, fcount_result = fcount(2,2)

        fcount_result_without_firstorder = sorted(fcount_result_without_firstorder)
        fcount_result = sorted(fcount_result)
        sample_result = sorted([[0,0],[0,1],[1,0],[2,0],[0,2],[1,1]])
        sample_result_without_firstorder = sorted([[0,0],[2,0],[0,2],[1,1]])
        # #
        # sample_result = sorted([(0,0),(0,1),(1,0),(2,0),(0,2),(1,1)])
        # sample_result_without_firstorder = sorted([(0,0),(2,0),(0,2),(1,1)])

        self.assertEqual(fcount_result_without_firstorder,sample_result_without_firstorder)
        self.assertEqual(fcount_result,sample_result)
    #

    def test_fcount_first_value_in_counter_should_be_zero(self):
        """
        Given that the code in raw_to_central.py is unchanged, and `ncounter.remove(counter[0])` line explicitly
        references to the zeroth element of the counter array (rather than by value of it), the `fcount` function
        should return zeros as the first element of its `counter` array.
        """

        test_parameters = itertools.product(range(1, 4), repeat=2)

        for params in test_parameters:
            answer = fcount(*params)[0][0]
            sum_of_answer = sum(answer)  # All zero's vector should have 0 sum
            self.assertEqual(sum_of_answer, 0, "The first element returned in counter list is not a zero vector, but: {0!r}".format(answer))
