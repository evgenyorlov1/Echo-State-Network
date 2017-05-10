from __future__ import division


import numpy as np

from utils.activation_functions import sigmoid_af


# TODO re implement harvesting function
def harvest_state(data, Vin, Wres, N, P):
    """
    Harvest reservoir states for input data.
    """
    X = np.zeros((P + 1, N))                                                 # reservoir node's states per input
    # Harvest reservoir states
    for i in xrange(1, P):
        for j in xrange(N):
            input_activation = data[0][i-1].dot(Vin[:, j])                 # activation from input
            recurrent_activation = X[i-1].dot(Wres[:, j])                  # activation from neurons
            X[i][j] = sigmoid_af(input_activation + recurrent_activation)
    X = np.delete(X, 0, axis=0)                                                 # delete zero state initial vector; P x N
    return X