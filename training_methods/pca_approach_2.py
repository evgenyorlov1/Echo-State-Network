from __future__ import division

from multiprocessing import Pool

from utils.activation_functions import sigmoid_af
from dimensionality_reduction_utils.PCA import pca_sklearn, pca_numpy, pca_numpy_R, pca_numpy_R_2

import numpy as np

from utils.norms import euclidean_norm


def __train(args):
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
        #print ' explained ration: {0}'.format( explained_ratio )
        reflection = I-Uk.dot(Uk.H)                                                 # N x N
        X[t] = reflection
    return X


def train_in_parallel(data, N, R, Lout, Washout, Vin, Wres):
    clusters = list()  # stores diff B for each l
    args = [
        [data[i], N, R, Washout, Vin, Wres] for i in xrange(Lout)
    ]
    pool = Pool(processes=Lout)
    clusters = pool.map(__train, args)
    return clusters


def classify(test_set, clusters, N, Lout, instances, Washout):
    T = len(test_set[0][0])  # number of pixel
    Y = np.zeros(instances)  # results of classification

    for i, image in enumerate(test_set[0][:instances]):
        result = dict()

        for l in xrange(Lout):
            response = np.zeros(N)
            res = 0
            for t in xrange(T):
                for _ in xrange(Washout):
                    response = __harvest_state(image[t], response)
                res += euclidean_norm(response.dot(clusters[l][t]))
            result[l] = res
        Y[i] = min(result, key=result.get)

    count = sum(1 for i, y in enumerate(Y) if y == test_set[1][i])
    print 'classify 2 count: {0}'.format(count)


def __harvest_state(u, previous_state, Vin, Wres):
    input_activation = Vin.dot(u)                                                   # activation from input
    recurrent_activation = previous_state.dot(Wres)                                 # activation from neurons
    X = sigmoid_af(input_activation + recurrent_activation)
    return X