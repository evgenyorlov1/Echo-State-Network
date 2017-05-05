from __future__ import division


from utils.activation_functions import sigmoid_af
from dimensionality_reduction_utils.PCA import pca_sklearn, pca_numpy, pca_numpy_R, pca_numpy_R_2

import numpy as np


def train(args):
    images = args[0]
    N = args[1]
    R = args[2]
    Washout = args[3]
    Vin = args[4]
    Wres = args[5]

    T = len(images[0])                                                              # pixels in image
    P = len(images)                                                                 # number of instances
    I = np.identity(N)

    X = np.zeros((N, P*T))                                                          # store image class
    for i, image in enumerate(images):
        for t in xrange(T):
            response = np.zeros(N)                                                  # initial response from reservoir
            for _ in xrange(Washout):
                response = __harvest_state(image[t], response, Vin, Wres)
            X[:, t + T*i] = response

    Uk, explained_ratio = pca_numpy_R_2(X, R, 0)                                    # N x R. Shit happens here with dimensions and exp ration
    reflection = I-Uk.dot(Uk.H)
    return reflection


def __harvest_state(u, previous_state, Vin, Wres):
    input_activation = Vin.dot(u)                                                   # activation from input
    recurrent_activation = previous_state.dot(Wres)                                 # activation from neurons
    X = sigmoid_af(input_activation + recurrent_activation)
    return X