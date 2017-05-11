# TODO convert to a class
from __future__ import division


from multiprocessing import Pool

from utils.activation_functions import sigmoid_af
from dimensionality_reduction_utils.PCA import pca_numpy
from utils.norms import euclidean_norm

import numpy as np


# TODO unpack arguments
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

    T = len(images[0])                                                              # pixels in image
    P = len(images)                                                                 # number of instances
    I = np.identity(N)

    X = np.zeros((T, N, N))

    for t in xrange(T):
        reservoir_responses = np.zeros((N, P))
        for _ in xrange(Washout):
            for p, image in enumerate(images):
                reservoir_responses[:, p] = harvest_state(image[t], reservoir_responses[:, p - 1], Vin, Wres)
        Uk, explained_ratio = pca_numpy(reservoir_responses, R, 0)
        assert Uk.shape == (N, R), 'Uk wrong shape'
        reflection = I - Uk.dot(Uk.T)
        assert reflection.shape == (N, N), 'reflection wrong'
        X[t] = reflection
        assert X[0].shape == (N, N), 'Wrong X sahpe'
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
    print 'Training'
    args = [
        [data[i], N, R, Washout, Vin, Wres] for i in xrange(Lout)
    ]
    pool = Pool(processes=Lout)
    clusters = pool.map(__train, args)
    assert clusters.__len__() == Lout, 'Wrong cluster size'
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
    T = len(test_set[0][0])
    Y = np.zeros(instances)

    for i, image in enumerate(test_set[0][:instances]):
        result = dict()
        res = 0
        for l in xrange(Lout):

            response = np.zeros(N)
            for _ in xrange(Washout - 1):
                for t in xrange(T):
                    response = harvest_state(image[t], response, Vin, Wres)

            for t in xrange(T):
                response = harvest_state(image[t], response, Vin, Wres)
                res += euclidean_norm(response.dot(clusters[l][t]))

            result[l] = res
        Y[i] = min(result, key=result.get)
    # TODO check count function
    count = sum(1 for i, y in enumerate(Y) if y == test_set[1][i])
    return count


def train_straight(train_set, N, R, P, Lout, Washout, Vin, Wres):
    clusters = list()
    T = len(train_set[0][0])  # pixels in image
    assert T == 784, "WRONG T.length"
    I = np.identity(N)
    for k in xrange(Lout):
        X = np.zeros((T, N, N))

        # TODO delete
#        Uk_temp = np.zeros((T, N))

        for t in xrange(T):
            reservoir_responses = np.zeros((N, P))  # saves responses from reservoir
            for p, image in enumerate(train_set[k]):
                response = np.zeros(N)  # initial response from reservoir
                for _ in xrange(Washout):
                    print image[t]
                    response = harvest_state(image[t], response, Vin, Wres)  # reservoir response; N
                reservoir_responses[:, p] = response
            Uk, explained_ratio = pca_numpy(reservoir_responses, R, 0)  # N x R
            # TODO delete
#            Uk_temp[t] = Uk.T
            reflection = I - Uk.dot(Uk.H)  # N x N
            assert I.shape == (N, N), 'Wrong reflection shape'
            X[t] = reflection
        assert X[0].shape == (N, N), 'Wrong X sahpe'
        clusters.append(X)         # can be mistake here
        assert clusters.__len__() == Lout, 'Wrong cluster size'
        # TODO delete
        #np.savetxt('Uk{0}'.format(k), Uk_temp)
    return clusters


# TODO move function to the base abstract class; separate harvesting from activation
def harvest_state(u, previous_state, Vin, Wres):
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
    X = sigmoid_af(input_activation + recurrent_activation)                         # 1 x N
    return X