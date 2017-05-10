import unittest
import numpy as np

from dimensionality_reduction_utils.PCA import *


class TestPCA(unittest.TestCase):
    def test_sort_eighenvalues(self):
        # equal size e, ev ndarray s
        e_test1 = np.array([0.0, 3.0, 1.3, 1.5, 1.9, 4.7, 2.3])
        ev_test1 = np.array([[1, 2, 3, 4, 5, 7, 9],
                           [11, 21, 31, 41, 51, 71, 91],
                           [12, 22, 32, 42, 52, 72, 92],
                           [13, 23, 33, 43, 53, 73, 93],
                           [14, 24, 34, 44, 54, 74, 94],
                           [15, 25, 35, 45, 55, 75, 95],
                           [16, 26, 36, 46, 56, 76, 96]])
        ev_test1_result = np.array([[1, 2, 3, 4, 5, 7, 9],
                             [11, 21, 31, 41, 51, 71, 91],
                             [12, 22, 32, 42, 52, 72, 92],
                             [13, 23, 33, 43, 53, 73, 93],
                             [14, 24, 34, 44, 54, 74, 94],
                             [15, 25, 35, 45, 55, 75, 95],
                             [16, 26, 36, 46, 56, 76, 96]])
        # test case: correct dimensions, function input
        e, ev = sort_eigenvalues(e_test1, ev_test1)
        print ev
        # test e sort in descending order
        #self.assertTrue((e == np.sort(e_test1)[::-1]).all(), 'e and e_test1.sorted are not equal. e is not sorted')
        self.assertTrue((ev == ev_test1_result).all(), 'ev and ev_test1.sorted are not equal. ev is not sorted according to e')


    def test_pca_sklearn(self):
        pass

    def test_pca_numpy(self):
        pass


if __name__ == '__main__':
    unittest.main()