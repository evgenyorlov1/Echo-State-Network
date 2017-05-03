from __future__ import division


from sklearn.decomposition import PCA
import numpy as np


def pca_sklearn(B, R):
    """
    Performs Principal Component Analysis.
    :param B: input matrix for dimension reduction
    :param accuracy: PCA accuracy, explained variance ratio
    :return: matrix with first n principal components, explained variance ratio. Reduce second dimension.
    """
    pca = PCA(n_components=R)
    print 'pca_sklearn R: {0}'.format(R)
    pca.fit(B)
    B = pca.transform(B)
    print 'sk_learn B shape: {0}'.format(B.shape)
    B = np.matrix(B)
    explained_variance_ratio = sum(pca.explained_variance_ratio_)               # explained varience ration after PCA
    return B, explained_variance_ratio


def pca_numpy(B, accuracy):
    """
    Principal Component Analysis via numpy.
    :param B: input matrix for dimension reduction
    :param accuracy: PCA accuracy, explained variance ratio
    :return: matrix with first n principal components, explained variance ratio
    """
    mean_value = np.mean(B, axis=0)
    A = B - mean_value
    C = np.cov(A, rowvar=1)
    e, ev = np.linalg.eig(C)                                                        # first e are important
    e = [i/sum(e) for i in e]
    explained_variance_ratio, count = 0, 0
    while explained_variance_ratio <= accuracy:
        explained_variance_ratio += e[count]
        count += 1

    new_feature = ev.T
    XTrans = new_feature.dot(A.T)
    return np.matrix(XTrans.T)[:, count], explained_variance_ratio


def pca_numpy_R(B, R, rowvar):
    """
    Principal Component Analysis via numpy.
    :param B: input matrix for dimension reduction
    :param R: first R principal components
    :param rowvar: If rowvar is 1, then each row represents a variable, with observations in the columns.
    :return: matrix with first n principal components, explained variance ratio
    """
    mean_value = np.mean(B, axis=0)
    A = B - mean_value
#    print 'normalized'
    C = np.cov(A, rowvar=rowvar)
#    print 'covariance shape: {0}'.format(C.shape)
    e, ev = np.linalg.eig(C)                                                        # first e are important
#    print 'eigen values shape: {0}'.format(ev.shape)
    e = [i/sum(e) for i in e]
    explained_variance_ratio = 0
#    print 'shape e length: {0}'.format(e.__len__())
    new_feature = ev.T
#    print 'new_feature shape: {0}'.format(new_feature.shape)
    XTrans = (new_feature.dot(A))
    XTrans = np.matrix(XTrans)
#    print 'R: {0}'.format(R)
    return XTrans[:, :R], explained_variance_ratio


#B = np.random.rand(100, 350500)
#print 'step 1'
#A, _ = pca_numpy_R(B, 300, 1)
#print 'A shape: {0}\n'.format(A.shape)

#C, _ = pca_sklearn(B, 300)
#print 'C shape: {0}'.format(C.shape)

