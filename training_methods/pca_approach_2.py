# TODO convert to a class
from __future__ import division


from multiprocessing import Pool
from joblib import Parallel, delayed

from utils.activation_functions import sigmoid_af
from dimensionality_reduction_utils.PCA import pca_numpy, pca_sklearn
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


def training(args):
    images = args[0]
    N = args[1]
    accuracy = args[2]
    Washout = args[3]
    Vin = args[4]
    Wres = args[5]

    T = len(images[0])  # pixels in image
    P = len(images)  # number of instances
    I = np.identity(N)

    X = np.zeros((T, N, P))
    Result = np.zeros((T, N, N))

    for i, image in enumerate(images):
        for _ in xrange(Washout):
            for t, pixel in enumerate(image):
                X[t, :, i] = neuron_activation(pixel, X[t - 1, :, i], Vin, Wres)

    # TODO delete
    x = np.zeros(T)
    for t in xrange(T):
        #print 'X[t].shape: {0}'.format(X[t].shape)
        Uk, expl, R = pca_numpy(X[t], accuracy, 0)  # TODO check 0/1
        temp, ex = pca_sklearn(X[t], 0.9)
        if temp.shape[1] > 1: print 'temp.shape: {0}; ex: {1}'.format(temp.shape, ex)
        if R > 1: print 'R: {0}; expl: {1}'.format(R, expl)
        # TODO delete
        x[t] = R
        Result[t] = I - Uk.dot(Uk.T)
    return Result, x


def train(data, N, accuracy, Lout, Washout, Vin, Wres):
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
        [data[i], N, accuracy, Washout, Vin, Wres] for i in xrange(Lout)
    ]
    pool = Pool(processes=Lout)
    clusters = pool.map(training, args)
    # TODO delete
    for i, cluster in enumerate(clusters):
        np.savetxt('dump_{0}'.format(i), cluster[1])
    return clusters


def classify(test_set, clusters, N, Lout, instances, Washout, Vin, Wres):
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
    Y = np.zeros(instances)
    count = 0
    for k, images in enumerate(test_set):
        for i, image in enumerate(images):                             # for image in images
            print 'i: {0}'.format(i)
            result = dict()
            _images = list()
            _images.append(image)
            Xtest = harvesting([_images, N, Washout, Vin, Wres])
            for l in xrange(Lout):                                                      # for class in classes
                res = 0
                X = clusters[l]
                for t, pixel in enumerate(image):
                    print 'X[t].shape: {0}; Xtest[t, :, 0].shape: {1}'.format(X[t].shape, Xtest[t, :, 0].shape)
                    res += euclidean_norm(X[t].dot(Xtest[t, :, 0]))
                result[l] = res
            Y[i] = min(result, key=result.get)


        count = sum(1 for y in Y if y == k)
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
