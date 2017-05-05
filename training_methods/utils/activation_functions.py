from __future__ import division


import numpy as np
import math


def sigmoid_af(x):
    """
    Sigmoid activation function for neuron.
    """
    return 1/(1+np.exp(-x))


def sin_af(x):
    return math.sin(x)


def cos_af(x):
    return math.cos(x)


def tanh_af(x):
    return math.tanh(x)