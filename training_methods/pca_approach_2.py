# TODO convert to a class
from __future__ import division


from multiprocessing import Pool

from utils.activation_functions import sigmoid_af
from dimensionality_reduction_utils.PCA import pca_numpy
from utils.norms import euclidean_norm

import numpy as np


# TODO unpack arguments
def harvesting(args):
    """
    Function that performs training in parallel by class.
    :param args: list contains arguments for function.
    :param args[0]: images. type numpy array
    :param args[1]: N. neurons
    :param args[2]: 
    :param args[3]:     
    :param args[4]:     
    :param args[5]:     
    :return: matrix ~ reflection of class.
    """
    images = args[0]
    N = args[1]
    Washout = args[2]
    Vin = args[3]
    Wres = args[4]

    T = len(images[0])                                                              # pixels in image
    P = len(images)                                                                 # number of instances

    X = np.zeros((T, N, P))

    for i, image in enumerate(images):
        for _ in xrange(Washout):
            for t, pixel in enumerate(image):
                X[t, :, i] = neuron_activation(pixel, X[t-1, :, i], Vin, Wres)

    return X


def train(data, N, Lout, Washout, Vin, Wres):
    """
    Parallel training for ESN pca approach 2.
    :param data: training data.
    :param N: number of neurons.
    :param R: number of first principal components.
    :param Lout: number of classes.
    :param Washout: number of washout times.
    :param Vin: input weight matrix for reservoir.
    :param Wres: reservoir weight matrix.
    :return: list of matrix for multiplication of signal to classify/
    """
    print 'Training'
    args = [
        [data[i], N, Washout, Vin, Wres] for i in xrange(Lout)
    ]
    pool = Pool(processes=Lout)
    clusters = pool.map(harvesting, args)
    return clusters


def classify(test_set, clusters, N, Lout, R, instances, Washout, Vin, Wres):
    """
    Classify test inputs.
    :param test_set: set for test classification.
    :param clusters: reflections of classes.
    :param N: number of neurons.
    :param Lout: number of classes.
    :param instances: number of instances to be classified.
    :param Washout: number of washout times.
    :param Vin: matrix of input reservoir weights.
    :param Wres: matrix of reservoir weights.
    :return: number of correct answers.
    """
    print 'Classify'
    T = len(test_set[0][0])
    Y = np.zeros(instances)
    I = np.identity(N)

    for i, image in enumerate(test_set[0][:instances]):                             # for image in images
        print 'i: {0}'.format(i)
        result = dict()
        images = list()
        images.append(image)
        Xtest = harvesting([images, N, Washout, Vin, Wres])
        print 'i: {0}; Xtest: done'.format(i)
        for l in xrange(Lout):                                                      # for class in classes
            print 'i: {0}; l: {1}'.format(i, l)
            res = 0
            X = clusters[l]
            for t, pixel in enumerate(image):
                Uk, _ = pca_numpy(X[t, :, :], R, 0)
                res += euclidean_norm((I - Uk.dot(Uk.T)).dot(Xtest[t, :, 0]))
            result[l] = res
        Y[i] = min(result, key=result.get)
    # TODO check count function
    count = sum(1 for i, y in enumerate(Y) if y == test_set[1][i])
    return count


def train_straight(train_set, N, R, P, Lout, Washout, Vin, Wres):
    pass


# TODO move function to the base abstract class; add feedback activation
def neuron_activation(u, previous_state, Vin, Wres):
    """
    Get reservoir response for a signal.
    :param u: input signal. Scalar or vector.
    :param previous_state: previous reservoir state.
    :param Vin: matrix of input reservoir weights.
    :param Wres: matrix of reservoir weights.
    :return: reservoir response.
    """
    input_activation = Vin.dot(u)
    assert input_activation.shape == Vin.shape, 'input activation wrong shape'
    recurrent_activation = previous_state.dot(Wres)                                 # activation from neurons
    X = sigmoid_af(input_activation + recurrent_activation)                         # K x N
    return X
