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
    X, Y, *_ = encode_data_with_pretrained(data_source, train_ds, test_ds, text_field, encoder_model, train_ds.examples + test_ds.examples, [])

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

        # make 4-part plot
        with plt.style.context('seaborn-whitegrid'):

            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['mathtext.fontset'] = 'dejavuserif'

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            axes = {"pretrained_elmo": ax1, "pretrained_glove": ax2, "sts_stsbenchmark": ax3, "sts_sick": ax4}
            model_names = {"pretrained_elmo": "ELMo", "pretrained_glove": "GloVe", 'sts_sick': "STS-sick", "sts_stsbenchmark": "STS-bench"}

            for embedding_model in ("pretrained_elmo", "pretrained_glove", "sts_stsbenchmark", "sts_sick"):
                X, Y, *_ = encode_data_with_pretrained(data_source, train_ds, test_ds, text_field, embedding_model, train_ds.examples + test_ds.examples, [])
                pca = PCA(n_components=n_components)
                pca_res = pca.fit_transform(X)
                tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
                tsne_pca_res = tsne.fit_transform(pca_res)
                ts1, ts2 = tsne_pca_res[:,0], tsne_pca_res[:,1]
                df_tsne_pca = DataFrame([{
                    'intent': intents[y],
                    'x-tsne-pca': t1,
                    'y-tsne-pca': t2,
                    'label': str(idx)
                } for idx, (y,t1,t2) in enumerate(zip(Y,ts1,ts2))])

                axis = axes[embedding_model]
                statistics = {}
                for idx, intent in enumerate(set(df_tsne_pca['intent'].values)):
                    values = [v[1:] for v in df_tsne_pca.loc[df_tsne_pca['intent']==intent].drop(columns=['intent']).values]
                    mean = np.mean(values, axis=0); var = np.var(values, axis=0)
                    statistics[intent] = (mean, var, f'C{idx}')
                    axis.scatter([v[0] for v in values], [v[1] for v in values], color=f'C{idx}', s=100, label=intent, alpha=0.4)
                    axis.add_patch(Ellipse(mean, var[0], var[1], fill=False, edgecolor=f'C{idx}', linewidth=2))
                    axis.set_title(model_names[embedding_model])

                for intent, stats in statistics.items():
                    axis.scatter([stats[0][0]], [stats[0][1]], color=stats[2], linewidth=3, s=500, marker='x')
                    for intent2, stats2 in statistics.items():
                        xs = [stats[0][0], stats2[0][0]]; ys = [stats[0][1], stats2[0][1]]
                        axis.plot(xs, ys, 'black', alpha=0.5, linewidth=2)

            handles, labels = ax4.get_legend_handles_labels()
            fig.legend(handles, labels, loc='center')
            fig.set_size_inches(15, 10)
            plt.savefig('./paper/dataset_viz4.pdf', format='pdf', dpi=100)
            plt.show()

        assert(False)
        
        pca = PCA(n_components=n_components)
        pca_res = pca.fit_transform(X)
        tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
        tsne_pca_res = tsne.fit_transform(pca_res)
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
