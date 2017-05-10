from __future__ import division


import numpy as np

from accuracy_measures.accuracy import accuracy
from utils.activation_functions import sigmoid_af


def classify(classify_set, Vin, Wres, Uout, N, Lout):
    P = len(classify_set[0])  # number of training elements
    X = harvest_states(classify_set, Vin, Wres, N, P)  # reservoir states
    Y = X.dot(Uout)  # NN output; P x Lout
    count = sum(1 for i in xrange(P) if Y[i][classify_set[1][i]] == max(Y[i]))
    return accuracy(P, count)


def train_straight(train_set, Vin, Wres, N, Lout):
    P = len(train_set[0])  # number of training elements
    X = harvest_states(train_set, Vin, Wres, N, P)  # reservoir states
    # Generate target vector
    y_target = np.zeros((P, Lout))  # target output
    for i in xrange(P):
        y_target[i][train_set[1][i]] = 1
    Uout, _, _, _ = np.linalg.lstsq(X, y_target)  # calculate U via linear regression


def harvest_states(data, Vin, Wres, N, P):
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