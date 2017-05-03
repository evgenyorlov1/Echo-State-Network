from __future__ import division


from utils.activation_functions import sigmoid_af
from dimensionality_reduction_utils.PCA import pca_sklearn, pca_numpy, pca_numpy_R

import numpy as np


def train(args):
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
        Uk, explained_ratio = pca_numpy_R(reservoir_responses, R, 1)                # N x R
        reflection = I-Uk.dot(Uk.H)                                                 # N x N
        X[t] = reflection
    return X


def __harvest_state(u, previous_state, Vin, Wres):
    input_activation = Vin.dot(u)                                                   # activation from input
    recurrent_activation = previous_state.dot(Wres)                                 # activation from neurons
    X = sigmoid_af(input_activation + recurrent_activation)
    return X