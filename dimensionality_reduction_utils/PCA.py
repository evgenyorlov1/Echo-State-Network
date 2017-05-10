from __future__ import division


from sklearn.decomposition import PCA
import numpy as np


def sort_eigenvalues(e, ev):
    """
    Sorts eighen vectors and values in a descending order (biggest first).
    :param e: eighen values
    :param ev: eighen vectors
    :return: eighen values, eighen vectors
    """
    indices = np.argsort(e)[::-1]                                                   # a[start:stop:step]
    e = e[indices]
    ev = ev[:, indices]
    return e, ev


def pca_sklearn(X, R):
    """
    Performs Principal Component Analysis. Reduce second dimension.
    :param X: input matrix for dimension reduction
    :param R: first R principal components
    :return: matrix with first n principal components, explained variance ratio.
    """
    pca = PCA(n_components=R)
    pca.fit(X)
    X = pca.transform(X)
    explained_variance_ratio = sum(pca.explained_variance_ratio_)                   # explained varience ration after PCA
    return X, explained_variance_ratio


def pca_numpy(X, R, rowvar):
    """
    Principal Component Analysis via numpy.
    :param X: input matrix for dimension reduction
    :param R: first R principal components
    :param rowvar: If rowvar is 1, then each row represents a variable, with observations in the columns
    :return: matrix with first n principal components (N x R), explained variance ratio
    """
    mean_value = np.mean(X, axis=rowvar)
    A = X - mean_value
    C = np.cov(A, rowvar=rowvar)
    e, ev = np.linalg.eigh(C)
    e, ev = sort_eigenvalues(e, ev)
    e = [i / sum(e) for i in e]
    explained_variance_ratio = sum((e[i]) for i in xrange(R))

    new_feature = ev.T
    XTrans = (new_feature.dot(A.T))
    XTrans = np.matrix(XTrans).T
    return XTrans[:, :R], explained_variance_ratio



