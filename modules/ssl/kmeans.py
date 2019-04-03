import numpy as np
import random

from collections import Counter
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

__all__ = ['kmeans']


def kmeans(X_l, X_u, Y_l, n_clusters):
    """
    K-mean clustering with majority label assignment.
    Args:
        X_l (list(emb)): embeddings of labeled data
        X_u (list(emb)): embeddings of unlabeled data
        Y_l (list(int)): labels of labeled data
        n_clusters (int): number of clusters (classes)
    Returns:
        Predicted labels for the unlabeled data
            (majority label from labeled data in their cluster).
    """
    nl = len(X_l)
    X = np.concatenate([X_l, X_u])
    all_predictions = []

    # repeat due to k-means random initialisation
    for n in range(5):
        clusterer = KMeans(n_clusters=n_clusters, n_init=10)
        predictions = clusterer.fit_predict(X)

        # get all true Y values for each cluster (from labeled examples)
        preds_labels = {i: [] for i in range(n_clusters)}
        for i, pred in enumerate(predictions[:nl]):
            preds_labels[pred].append(Y_l[i])
        
        # label each cluster (majority label of labeled points in that cluster)
        cluster_labels = {}
        for i in range(n_clusters):
            if len(preds_labels[i]) == 0:
                cluster_label = random.randint(0,n_clusters-1)
            else:
                cluster_label, _ = Counter(preds_labels[i]).most_common(1)[0]
            cluster_labels[i] = cluster_label
        
        predictions = [cluster_labels[c_idx] for c_idx in predictions]
        all_predictions.append(predictions)
    
    # take most common label over 5 trials for each unlabeled point
    predictions = mode(np.array(all_predictions), axis=0).mode[0][nl:]

    return predictions


def recursive_kmeans(x_l, x_u, y_l, n_clusters):
    nl = len(x_l)
    X = np.concatenate([X_l, X_u])
    all_predictions = []
    indices = []

    clusterer = KMeans(n_clusters=n_clusters, n_init=10)
    clusters = {}
    while True:
        preds = clusterer.fit_predict(X)
        # for i, cluster in enumerate(preds):
