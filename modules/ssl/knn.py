def warn(*args, **kwargs): # suppress sklearn invalid metric warning.
    pass
import warnings
warnings.warn = warn

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

__all__ = ['knn_classify']


def knn_classify(xs_l, ys_l, xs_u, n=1, weights='uniform', distance_metric='euclidean', threshold=None):
    """
    Fits a KNN model to labeled data and output labels for unlabeled data.
    Args:
        n: number of neighbors for KNN
        xs_l (np.ndarray): embeddings for labeled data
        ys_l (np.ndarray): labels for labeled data
        xs_u (np.ndarray): embeddings for unlabeled data
        ys_u (np.ndarray): labels for unlabeled data (not shown on purpose)
        weights (str): KNN argument
        distance_metric: (str): KNN argument
    Returns:
        Classifications for the unlabeled data, and indices of the unlabeled data points that have
            been labelled.
    """
    knn_model = KNeighborsClassifier(n_neighbors=n, weights=weights, algorithm='auto', metric=distance_metric)
    knn_model.fit(xs_l, ys_l)

    if threshold:
        classifications, indices = [], []
        for i, prob in enumerate(knn_model.predict_proba(xs_u)):
            if np.max(prob) > threshold:
                classifications.append(np.argmax(prob))
                indices.append(i)
    else:
        classifications = knn_model.predict(xs_u)
        indices = [i for i in range(len(xs_u))]

    return classifications, indices
