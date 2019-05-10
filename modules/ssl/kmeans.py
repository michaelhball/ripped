import numpy as np
import random

from collections import Counter
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

__all__ = ['kmeans', 'recursive_kmeans']


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
    """
    takes in some data, returns labels for all unlabeled points (labels corresponding to actual data labels, not cluster ids)
    """
    X = np.concatenate([x_l, x_u])
    clusterer = KMeans(n_clusters=n_clusters, n_init=10)
    preds = clusterer.fit_predict(X)

    # idxes of data points in each cluster
    clusters = {i: [] for i in range(n_clusters)}
    for idx, cluster in enumerate(preds):
        clusters[cluster].append(idx)
    
    # list of labels from labeled points in each cluster
    cluster_label_lists = {i: [] for i in range(n_clusters)}
    for cluster_ind, cluster_idxes in clusters.items():
        for idx in cluster_idxes:
            if idx < len(y_l):
                cluster_label_lists[cluster_ind].append(y_l[idx])

    # label unlabeled points in each cluster
    y_u = np.array([-1 for _ in range(len(x_u))])
    for cluster_ind, label_list in cluster_label_lists.items():
        cluster_idxes = clusters[cluster_ind]
        if len(set(label_list)) > 1:
            n_clusters_recur = len(set(label_list))
            x_l_recur, x_u_recur, y_l_recur = [], [], []
            x_u_recur_idxes = [] # indices of unlabeled points were recurring on
            for idx in cluster_idxes:
                if idx < len(x_l):
                    x_l_recur.append(x_l[idx])
                    y_l_recur.append(y_l[idx])
                else:
                    x_u_recur.append(x_u[idx-len(x_l)])
                    x_u_recur_idxes.append(idx-len(x_l))
            if len(x_u_recur) > 0:
                x_u_recur_labels = recursive_kmeans(x_l_recur, x_u_recur, y_l_recur, n_clusters_recur)
                y_u[x_u_recur_idxes] = x_u_recur_labels
        else:
            # CHECK IF THIS RANDOM LABELLING IS THE DESIRED BEHAVIOUR WHEN WE HAVE NO LABELED EXAMPLES IN A CLUSTER
            cluster_label = random.randint(0,n_clusters-1) if len(label_list) == 0 else label_list[0]
            for idx in cluster_idxes:
                if idx >= len(x_l):
                    y_u[idx-len(x_l)] = cluster_label

    assert(-1 not in y_u)
    return y_u
