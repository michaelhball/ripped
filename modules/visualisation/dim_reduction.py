from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from modules.utilities.imports import *
from modules.ssl import encode_data_with_pretrained

__all__ = ['visualise_data']


def visualise_data(data_source, encoder_model, datasets, intents, text_field, type_='pca', show=True):
    train_ds, val_ds, test_ds = datasets
    if encoder_model.startswith('pretrained'):
        embedding_type = encoder_model.split('_')[1]
    elif encoder_model.startswith('sts'):
        embedding_type = encoder_model

    X, Y, *_ = encode_data_with_pretrained(data_source, train_ds, test_ds, text_field, embedding_type, train_ds.examples + test_ds.examples, [])

    nc = len(set(Y))
    intent_embeddings = {i: [] for i in range(nc)}
    for x,y in zip(X,Y):
        intent_embeddings[y].append(x)
    intent_stats = {}
    for intent, embeddings in intent_embeddings.items():
        intent_stats[intent] = (np.mean(embeddings, axis=0), np.var(embeddings, axis=0))
    
    cluster_distances = []
    for i1, stats in intent_stats.items():
        max_dist = 0
        for i2, other_stats in intent_stats.items():
            dist = np.linalg.norm(np.abs(stats[0]-other_stats[0]), axis=0)
            if dist > max_dist:
                max_dist = dist
        cluster_distances.append(np.linalg.norm(stats[1]) / max_dist)
    print(f'{np.mean(cluster_distances)}\t{np.max(cluster_distances)}\t{np.min(cluster_distances)}\t{np.mean([s[1] for s in intent_stats.values()])}')

    if type_ == "pca":
        pca = PCA(n_components=3)
        pca_res = pca.fit_transform(X)
        pcs1, pcs2, pcs3 = pca_res[:,0], pca_res[:,1], pca_res[:,2]
        print(f'Explained variation per principal component: {pca.explained_variance_ratio_}')
        df_pca = DataFrame([{'principle component 1': pc1, 'principle component 2': pc2, 'intent': intents[y]} for y,pc1,pc2 in zip(Y,pcs1,pcs2)])
    
    elif type_ == "tsne":
        tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
        tsne_res = tsne.fit_transform(X)
        tsnes1, tsnes2 = tsne_res[:,0], tsne_res[:,1]
        df_tsne = DataFrame([{'x-tsne': t1, 'y-tsne': t2, 'intent': intents[y]} for y,t1,t2 in zip(Y,tsnes1,tsnes2)])
 
    elif type_ == "pca+tsne":
        n_components = {'webapps': 30, 'chatbot': 50, 'askubuntu': 50, 'chat': 100}[data_source]
        pca = PCA(n_components=n_components)
        pca_res = pca.fit_transform(X)
        # print(f'Cumulative explained variation for {n_components} principal components: {np.sum(pca.explained_variance_ratio_)}')
        tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
        tsne_pca_res = tsne.fit_transform(pca_res)
        # tsne_pca_res = tsne.fit_transform(X)
        ts1, ts2 = tsne_pca_res[:,0], tsne_pca_res[:,1]
        df_tsne_pca = DataFrame([{
            'intent': intents[y],
            'x-tsne-pca': t1,
            'y-tsne-pca': t2,
            'label': str(idx)
        } for idx, (y,t1,t2) in enumerate(zip(Y,ts1,ts2))])

        # plot and calculate variance/inf(means)
        with plt.style.context('seaborn-whitegrid'):
            plt.figure()
            ax = plt.gca()
            
            intent_stats = {}
            for idx, intent in enumerate(set(df_tsne_pca['intent'].values)):
                values = [v[1:] for v in df_tsne_pca.loc[df_tsne_pca['intent']==intent].drop(columns=['intent']).values]
                mean = np.mean(values, axis=0); var = np.var(values, axis=0)
                intent_stats[intent] = (mean, np.var(values))
                ax.scatter([v[0] for v in values], [v[1] for v in values], label=intent, color=f'C{idx}', s=100)
                ax.add_patch(Ellipse(mean, np.sqrt(var[0]), np.sqrt(var[1]), fill=False, edgecolor=f'C{idx}', linewidth=2)) # sqrt here makes variance-->std
            
            if show:
                plt.xlabel('tsne-pca-1')
                plt.ylabel('tsne-pca-2')
                plt.title(f'PCA + tSNE embeddings. Dataset: {data_source}, embedding method: {embedding_type}')
                plt.legend()
                plt.show()

            assert(False)

    else:
        print(f"'{type_}'dimensionality reduction is not supported")
        return
