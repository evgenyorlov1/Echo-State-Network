from __future__ import division

from multiprocessing import Pool

from utils.activation_functions import sigmoid_af
from dimensionality_reduction_utils.PCA import pca_sklearn, pca_numpy, pca_numpy_R, pca_numpy_R_2

import numpy as np

from utils.norms import euclidean_norm


def __train(args):
    """
    Function that performs training in parallel by class.
    :param args: list contains arguments for function.
    :return: matrix ~ reflection of class.
    """
    images = args[0]
    N = args[1]
    R = args[2]
    Washout = args[3]
    Vin = args[4]
    Wres = args[5]

    T = len(images[0])  # pixels in image
    P = len(images)     # number of instances
    I = np.identity(N)

    X = np.zeros((T, N, N))
    for t in xrange(T):
        reservoir_responses = np.zeros((N, P))                                      # saves responses from reservoir
        response = np.zeros(N)                                                      # initial response from reservoir
        for p, image in enumerate(images):
            for _ in xrange(Washout):
                response = __harvest_state(image[t], response, Vin, Wres)           # reservoir response; N
            reservoir_responses[:, p] = response

        Uk, explained_ratio = pca_numpy_R_2(reservoir_responses, R, 0)                # N x R
        reflection = I-Uk.dot(Uk.H)                                                 # N x N
        X[t] = reflection
    return X


def train_in_parallel(data, N, R, Lout, Washout, Vin, Wres):
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
    args = [
        [data[i], N, R, Washout, Vin, Wres] for i in xrange(Lout)
    ]
    pool = Pool(processes=Lout)
    clusters = pool.map(__train, args)
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
    T = len(test_set[0][0])  # number of pixel
    Y = np.zeros(instances)  # results of classification

    for i, image in enumerate(test_set[0][:instances]):
        result = dict()

        for l in xrange(Lout):
            response = np.zeros(N)
            res = 0
            for t in xrange(T):
                for _ in xrange(Washout):
                    response = __harvest_state(image[t], response, Vin, Wres)
                res += euclidean_norm(response.dot(clusters[l][t]))
            result[l] = res
        Y[i] = min(result, key=result.get)

    count = sum(1 for i, y in enumerate(Y) if y == test_set[1][i])
    return count


def train_straight(train_set, N, R, Lout, Washout, Vin, Wres):
    clusters = list()
    T = len(train_set[0][0])  # pixels in image
    I = np.identity(N)
    for k in xrange(Lout):
        X = np.zeros((T, N, N))
        for t in xrange(T):
            reservoir_responses = np.zeros((N, P))  # saves responses from reservoir
            response = np.zeros(N)  # initial response from reservoir
            for p, image in enumerate(data[k]):
                for _ in xrange(Washout):
                    response = __harvest_state(image[t], response, Vin, Wres)  # reservoir response; N
                reservoir_responses[:, p] = response
            Uk, explained_ratio = pca_numpy_R_2(reservoir_responses, R, 0)  # N x R
            # print 'Uk.shape: {0}'.format(Uk.shape)
            reflection = I - Uk.dot(Uk.H)  # N x N
            # if np.iscomplex(reflection.all()): print 'reflection is complex'
            X[t] = reflection
        clusters.append(X)
    return clusters


def __harvest_state(u, previous_state, Vin, Wres):
    """
    Get reservoir response for a signal.
    :param u: input signal.
    :param previous_state: previous reservoir state.
    :param Vin: matrix of input reservoir weights.
    :param Wres: matrix of reservoir weights.
    :return: reservoir response.
    """
    input_activation = Vin.dot(u)                                                   # activation from input
    recurrent_activation = previous_state.dot(Wres)                                 # activation from neurons
    X = sigmoid_af(input_activation + recurrent_activation)
    return X