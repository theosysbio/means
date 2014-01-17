from fcount import fcount
import unittest

class FcountTestCase(unittest.TestCase):
    def test_fcount_2_2(self):
        fcount_result_without_firstorder, fcount_result = fcount(2,2)
        fcount_result_without_firstorder = sorted(fcount_result_without_firstorder)
        fcount_result = sorted(fcount_result)
        sample_result = sorted([[0,0],[0,1],[1,0],[2,0],[0,2],[1,1]])
        sample_result_without_firstorder = sorted([[0,0],[2,0],[0,2],[1,1]])
        self.assertEqual(fcount_result_without_firstorder,sample_result_without_firstorder)
        self.assertEqual(fcount_result,sample_result)