import numpy as np

from sklearn.neighbors import KNeighborsClassifier


def knn_classify(n, xs_l, ys_l, xs_u, ys_u=None, weights='uniform', distance_metric='euclidean'):
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
        Classifications for the unlabeled data, and accuracy of these
            classifications if unlabeled labels are given.
    """
    classifier = KNeighborsClassifier(n_neighbors=n, weights=weights, algorithm='auto', metric=distance_metric)
    classifier.fit(xs_l, ys_l)
    classifications = classifier.predict(xs_u)
    if not ys_u is None:
        num_correct = np.sum(classifications == ys_u)
    return classifications, num_correct


# Make another function for the recursive version, only giving labels if confidence is high enough