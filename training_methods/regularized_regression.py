from accuracy_measures.accuracy import accuracy
import numpy as np
import sys


# TODO switch to regularized regression
def train_for_regularized_least_squares(train_set, Lout):
    """
    Trains ESN on dataset. Calculates U via least square linear regression.
    """
    T = len(train_set[0])                                                           # number of training elements
    X = __harvest_states(train_set, T, '3/4. Training:')                            # reservoir states
    # Generate target vector
    y_target = np.zeros((T, Lout))                                                  # target output
    for i in xrange(T):
        y_target[i][train_set[1][i]] = 1
    print '{0: <45}'.format('Calculation output matrix...'),
    sys.stdout.flush()
    Uout, _, _, _ = np.linalg.lstsq(X, y_target)                                    # calculate U via linear regression
    print '{0: <45}'.format('OK')
    return Uout

# TESTED just for readout classification
def classify_for_regularized_least_squares(valid_set, Uout):
    """
    Classifies input information via output nodes training with least squares regression.
    """
    T = len(valid_set[0])                                                           # number of training elements
    X = __harvest_states(valid_set, T, '4/4. Classifying:')                         # reservoir states
    print '{0: <45}'.format('4/4. Classifying ESN...'),
    sys.stdout.flush()
    Y = X.dot(Uout)                                                                 # NN output
    count = sum(1 for i in xrange(T) if Y[i][valid_set[1][i]] == max(Y[i]))
    print '{0: <45}'.format('OK')
    print '{0: <45}'.format('4/4. Classification accuracy measures...'),
    sys.stdout.flush()
    print '{0: <45}'.format('OK')
    return accuracy(T, count), Y