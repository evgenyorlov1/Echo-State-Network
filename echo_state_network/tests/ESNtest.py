import gzip
import array
import os
import unittest

import cPickle

from echo_state_network.ESN import *


class ESNtest(unittest.TestCase):

    def setUp(self):
        file = '/home/evgenyorlov1/Echo-State-Network/datasets/mnist.pkl.gz'
        neurons = 100
        alfa = 0.5
        sparsity = 0.2
        R = 10
        w = 4
        self.esn = ESN(file, neurons, alfa, sparsity, R, w)
        self.esn.load_dataset()
        self.esn.initialize()

    def test__ESN__harvest_states(self):
        pass

    def test__ESN__harvest_state(self):
        pass

    def test__ESN__c_indices(self):
        indices = dict()
        for i in xrange(self.esn.Lout):
            arr = array.array('i')
            for j in xrange(len(self.esn.train_set[0])):
                if self.esn.train_set[1][j] == i:
                    arr.append(j)
            indices[i] = np.array(arr)

        self.assertTrue(((self.esn.C_indices[i] == indices[i]).all() for i in xrange(len(indices))))

    def test__ESN__data_by_class(self):
        pass


if __name__ == '__main__':
    unittest.main()