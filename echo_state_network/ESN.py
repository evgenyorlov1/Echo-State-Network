from __future__ import division

from multiprocessing import Pool
from accuracy_measures.accuracy import accuracy
from utils.norms import euclidean_norm
from utils.activation_functions import sigmoid_af
from training_methods.pca_approach_1 import train as train_1
from training_methods.pca_approach_2 import __train as train_2, \
    classify as classify_2, \
    train_in_parallel as train_in_parallel_2, \
    train_straight as train_straight_2
from training_methods.pca_approach_3 import train as train_3
from dimensionality_reduction_utils.PCA import pca_numpy

import progressbar as pb
import numpy as np
import gzip
import cPickle
import sys
import os


class ESN:
    """
    Echo State Network implementation.

    Attributes:
        file                file with data
        train_set           set for ESN training
        valid_set           set for ESN validation
        test_set            set for ESN classification
        labels              set of class labels for train set
        C_indices           indices per class
        Washout             washout period
        N                   number of neurons in reservoir
        Kin                 number of input neurons
        Lout                number of output neurons
        sparsity            sparsity/connectivity of RNN; number of non zero weights
        Vin                 matrix with weights for input neurons; KxN
        Wres                matrix with weights for reservoir neurons; NxN
        Uout                matrix with weights for output neurons; NxL
        X                   matrix with reservoir states for classification; TxN
        B                   list of matrices with normalized 1D points, for each class respectively
    """

    def __init__(self, file, neurons, alfa, sparsity, R, w):
        self.file      =    file
        self.train_set =    None
        self.valid_set =    None
        self.test_set  =    None
        self.labels    =    None
        self.C_indices =    dict()
        self.Washout   =    w
        self.N         =    neurons
        self.Kin       =    None
        self.Lout      =    None
        self.alfa      =    alfa
        self.sparsity  =    sparsity
        self.Vin       =    None
        self.Wres      =    None
        self.Uout      =    None
        self.X         =    None
        self.clusters  =    None
        self.R         =    R

    # ADD feedback from output
    def __harvest_states(self, data, P, label):
        """
        Harvest reservoir states for input data.
        """
        X = np.zeros((P + 1, self.N))                                                 # reservoir node's states per input
        label = '{0: <13}'.format(label)
        # Harvest reservoir states
        bar = pb.ProgressBar(maxval=P, widgets=[label, pb.Bar('=', '[', ']'), ' ', pb.Percentage()]).start()
        for i in xrange(1, P):
            for j in xrange(self.N):
                input_activation = data[0][i-1].dot(self.Vin[:, j])                 # activation from input
                recurrent_activation = X[i-1].dot(self.Wres[:, j])                  # activation from neurons
                X[i][j] = sigmoid_af(input_activation + recurrent_activation)
            bar.update(i)
        bar.finish()
        X = np.delete(X, 0, axis=0)                                                 # delete zero state initial vector; P x N
        return X

    # ADD feedback from output
    def __harvest_state(self, u, previous_state):
        input_activation = self.Vin.dot(u)                                          # activation from input
        recurrent_activation = previous_state.dot(self.Wres)                        # activation from neurons
        X = sigmoid_af(input_activation + recurrent_activation)
        return X

    def __c_indices(self):
        """
        Set indices per class.
        """
        for i in xrange(self.Lout):                                                    # get indices per class {class:indices}
            self.C_indices[i] = np.where(self.labels == i)[0]

    def __data_by_class(self, P):
        """
        Returns data separated by class. Only train data without class label. 
        :param P: number of train instances to select
        :return: list with train inputs by class
        """
        data = list()                                                               # stores train inputs for each class
        T = len(self.train_set[0][0])                                               # pixels in image
        for i in xrange(self.Lout):
            indices = self.C_indices[i][:P]
            images = np.zeros((P, T))
            for k, j in enumerate(indices):
                images[k] = self.train_set[0][j]
            data.append(images)
        return data

    def load_dataset(self):
        """
        Loads MNIST.
        Each of the three lists is a pair formed from a list of images and a list of class labels for each of the images.
        An image is represented as numpy 1-dimensional array of 784 (28 x 28) float values between 0 and 1
        (0 stands for black, 1 for white). The labels are numbers between 0 and 9 indicating which digit the image represents.
        """
        print '{0: <45}'.format('1/4. Loading dataset...'),
        sys.stdout.flush()
        with gzip.open(self.file, 'rb') as f:
            self.train_set, self.valid_set, self.test_set = cPickle.load(f)
        _, self.labels = self.train_set
        print '{0: <45}'.format('OK')

    # Mila said to re implement. Ask her how
    def initialize(self):
        """
        Initialize ESN: K, L, V, W.
        """
        print '{0: <45}'.format('2/4. Initializing Echo State Network...'),
        sys.stdout.flush()
        # TODO fix seeds for testing
        np.random.seed(None)
        self.Kin = 1#len(self.train_set[0][1])                                          # define number of input neurons;  1 # just for PCA#
        self.Lout = max(self.train_set[1]) + 1                                         # define number of output neurons
        self.Vin = np.random.uniform(-self.alfa, self.alfa, (self.Kin, self.N))       # init weights for input neurons
        self.Wres = np.random.uniform(-self.alfa, self.alfa, (self.N, self.N))      # init weights for reservoir neurons
        self.Uout = np.random.uniform(-self.alfa, self.alfa, (self.N, self.Lout))      # init weights for output neurons. Need for pca clustering
        # Sparsity
        mask = (np.random.uniform(size=(self.N, self.N)) < (1-self.sparsity))       # sparsity mask
        self.Wres[mask] = 0.0
        self.__c_indices()
        print '{0: <45}'.format('OK')

    def train_for_regularized_least_squares(self):
        """
        Trains ESN on dataset. Calculates U via least square linear regression.
        """
        P = len(self.train_set[0])                                                  # number of training elements
        X = self.__harvest_states(self.train_set, P, '3/4. Training:')              # reservoir states
        # Generate target vector
        y_target = np.zeros((P, self.Lout))                                         # target output
        for i in xrange(P):
            y_target[i][self.train_set[1][i]] = 1
        print '{0: <45}'.format('Calculation output matrix...'),
        sys.stdout.flush()
        self.Uout, _, _, _ = np.linalg.lstsq(X, y_target)                           # calculate U via linear regression
        print '{0: <45}'.format('OK')

    def classify_for_regularized_least_squares(self):
        """
        Classifies input information via output nodes training with least squares regression.
        """
        P = len(self.valid_set[0])                                                  # number of training elements
        self.X = self.__harvest_states(self.valid_set, P, '4/4. Classifying:')      # reservoir states
        print '{0: <45}'.format('4/4. Classifying ESN...'),
        sys.stdout.flush()
        Y = self.X.dot(self.Uout)                                                   # NN output; P x Lout
        count = sum(1 for i in xrange(P) if Y[i][self.valid_set[1][i]] == max(Y[i]))
        print '{0: <45}'.format('OK')
        print '{0: <45}'.format('4/4. Classification accuracy measures...'),
        sys.stdout.flush()
        print '{0: <45}'.format('OK')
        return accuracy(P, count)

    # TODO REFACTOR B; possible input for U
    def train_for_clustering_with_principal_components_approach1(self):
        """
        Trains ESN on dataset. input information via output nodes training with PCA.
        """
        self.clusters = list()                                                      # stores diff B for each l
        P = 200                                                                     # number of instances
        data = self.__data_by_class(P)
        args = [
            [data[i], self.N, self.R, self.Washout, self.Vin, self.Wres] for i in xrange(self.Lout)
        ]
        pool = Pool(processes=self.Lout)
        self.clusters = pool.map(train_1, args)

    def classify_for_clustering_with_principal_components_approach1(self):
        """
        Classifies input information via PCA decomposition.
        """
        P = 100                                                                     # number of instances
        T = len(self.valid_set[0][0])                                               # number of training elements
        Y = np.zeros(P)
        for i, image in enumerate(self.valid_set[0][:P]):
            result = dict()

            b = np.zeros(T)                                                         # l-2 norms for image
            response = np.zeros(self.N)
            for t, point in enumerate(image):
                for _ in xrange(self.Washout):
                    response = self.__harvest_state(point, response)
                    b[t] = euclidean_norm(response)

            for l in xrange(self.Lout):
                result[l] = euclidean_norm(b.dot(self.clusters[l]))
            Y[i] = min(result, key=result.get)
        count = sum(1 for i, y in enumerate(Y) if y == self.valid_set[1][i])
        return accuracy(P, count)

    # TODO test
    def train_for_clustering_with_principal_components_approach2(self):
        self.clusters = list()                                                      # stores diff B for each l                                                     # identity matrix
        P = 600                                                                      # number of instances

        data = self.__data_by_class(P)

        self.clusters = train_straight_2(data, self.N, self.R, P, self.Lout, self.Washout, self.Vin, self.Wres)

    # TODO test
    def train_for_clustering_with_principal_components_approach2_paralel(self):
        self.clusters = list()                                                      # stores diff B for each l
        P = 30                                                                      # number of instances
        data = self.__data_by_class(P)
        self.clusters = train_in_parallel_2(data,
                                            self.N,
                                            self.R,
                                            self.Lout,
                                            self.Washout,
                                            self.Vin,
                                            self.Wres)
        print 'Finish!'

    # TODO test
    def classify_for_clustering_with_principal_components_approach2(self):
        instances = 100                                                                     # number of instances
        count = classify_2(self.valid_set, self.clusters, self.N, self.Lout, instances, self.Washout, self.Vin, self.Wres)
        print 'classify 2 count: {0}'.format(count)
        return accuracy(instances, count)

    # TODO
    def train_for_clustering_with_principal_components_approach3(self):
        self.clusters = list()                                                      # stores (I - Uk*Uk.H) for each k
        T = len(self.train_set[0][0])                                               # pixels in image
        I = np.identity(self.N)                                                     # identity matrix
        P = 30                                                                      # number of instances

        data = self.__data_by_class(P)

        for k in xrange(self.Lout):
            print 'class: {0}'.format(k)
            X = np.zeros((self.N, P*T))                                             # store image class

            for i, image in enumerate(data[k]):
                for t in xrange(T):
                    response = np.zeros(self.N)                                     # initial response from reservoir
                    for _ in xrange(self.Washout):
                        response = self.__harvest_state(image[t], response)
                    X[:, t + T*i] = response
            print 'X.shape: {0}'.format(X.shape)
            Uk, explained_ratio = pca_numpy(X, self.R, 0)                            # N x R. Shit happens here with dimensions and exp ration
            reflection = I-Uk.dot(Uk.H)                                             # N x N
            self.clusters.append(reflection)

    def train_for_clustering_with_principal_components_approach3_paralel(self):
        self.clusters = list()                                                      # stores diff B for each l
        P = 30                                                                     # number of instances

        data = self.__data_by_class(P)

        args = [
            [data[i], self.N, self.R, self.Washout, self.Vin, self.Wres] for i in xrange(self.Lout)
        ]
        pool = Pool(processes=self.Lout)
        self.clusters = pool.map(train_3, args)
        print 'Finish!'

    # TODO
    def classify_for_clustering_with_principal_components_approach3(self):
        instances = 300                                                                     # number of instances
        T = len(self.valid_set[0][0])                                               # number of pixels
        Y = np.zeros(instances)                                                             # results of classification

        for i, image in enumerate(self.valid_set[0][:instances]):
            result = dict()
                                                                                                                                                                                                            
            for l in xrange(self.Lout):
                X = np.zeros((self.N, T))                                           # input to classify
                response = np.zeros(self.N)
                for t in xrange(T):
                    for _ in xrange(self.Washout):
                        response = self.__harvest_state(image[t], response)
                    X[:, t] = response
                result[l] = euclidean_norm(self.clusters[l].dot(X))

            Y[i] = min(result, key=result.get)
        count = sum(1 for i, y in enumerate(Y) if y == self.valid_set[1][i])
        print 'approach 3 count: {0}'.format(count)
        return accuracy(instances, count)
