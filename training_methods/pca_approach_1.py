from __future__ import division


from utils.activation_functions import sigmoid_af
from dimensionality_reduction_utils.PCA import pca_numpy
from utils.norms import euclidean_norm

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
    I = np.identity(T)

    B = np.zeros((T, P))  # B cluster T x P, stores bi
    for p, image in enumerate(images):
        b = np.zeros(T)                                                             # l-2 norms for image
        response = np.zeros(N)
        for t, point in enumerate(image):
            for _ in xrange(Washout):
                response = __harvest_state(point, response, Vin, Wres)
                b[t] = euclidean_norm(response)
        B[:, p] = b

        Uk, _ = pca_numpy(B, R, 0)  # T x R
        reflection = I-Uk.dot(Uk.H)  # T x T
        return reflection  # T x T


def __harvest_state(u, previous_state, Vin, Wres):
    input_activation = Vin.dot(u)                                                   # activation from input
    recurrent_activation = previous_state.dot(Wres)                                 # activation from neurons
    X = sigmoid_af(input_activation + recurrent_activation)
    return X
