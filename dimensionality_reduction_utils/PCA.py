from __future__ import division


from sklearn.decomposition import PCA
import numpy as np


def __sort_eighenvalues(e, ev):
    indices = np.argsort(e)[::-1]
    e = e[indices]
    ev = ev[:, indices]
    return e, ev


def pca_sklearn(B, R):
    """
    Performs Principal Component Analysis.
    :param B: input matrix for dimension reduction
    :param accuracy: PCA accuracy, explained variance ratio
    :return: matrix with first n principal components, explained variance ratio. Reduce second dimension.
    """
    pca = PCA(n_components=R)
    pca.fit(B)
    B = pca.transform(B)
    B = np.matrix(B)
    explained_variance_ratio = sum(pca.explained_variance_ratio_)               # explained varience ration after PCA
    return B, explained_variance_ratio


def pca_numpy_R_2(B, R, rowvar):
    """
    Principal Component Analysis via numpy.
    :param B: input matrix for dimension reduction.
    :param R: first R principal components.
    :param rowvar: If rowvar is 1, then each row represents a variable, with observations in the columns.
    :return: matrix with first n principal components, explained variance ratio.
    """
    mean_value = np.mean(B, axis=0)
    A = B - mean_value
    C = np.cov(A, rowvar=rowvar)
    e, ev = np.linalg.eigh(C)
    e, ev = __sort_eighenvalues(e, ev)
    e = [i / sum(e) for i in e]
    explained_variance_ratio = sum((e[i]) for i in xrange(R))

    new_feature = ev.T
    XTrans = (new_feature.dot(A.T))
    XTrans = np.matrix(XTrans).T
    return XTrans[:, :R], explained_variance_ratio
