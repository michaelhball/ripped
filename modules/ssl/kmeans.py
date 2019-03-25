from collections import Counter
from sklearn.cluster import KMeans

__all__ = ['kmeans_accuracy']


def kmeans_accuracy(X, Y, n_clusters):
    clusterer = KMeans(n_clusters=n_clusters, n_init=10)
    predictions = clusterer.fit_predict(X)
    preds_labels = {i: [] for i in range(n_clusters)}
    for i, pred in enumerate(predictions):
        preds_labels[pred].append(Y[i])

    ys_seen, total_correct = [], 0
    for i in range(n_clusters):
        y, num_correct = Counter(preds_labels[i]).most_common(1)[0]
        ys_seen.append(y); total_correct += num_correct
        print(y, num_correct, len(preds_labels[i]))
    print(ys_seen)
    print(float(total_correct/len(Y)))
