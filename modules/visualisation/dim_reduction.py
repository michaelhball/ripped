from pandas import DataFrame
from plotnine import *
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

    X, Y, *_ = encode_data_with_pretrained(data_source, train_ds, text_field, embedding_type,  train_ds.examples, [])
    X = np.array([x / np.linalg.norm(x) for x in X]) # not sure if we need normalisation?

    print(len(X[0]))

    if type_ == "pca":
        pca = PCA(n_components=3)
        pca_res = pca.fit_transform(X)
        pcs1, pcs2, pcs3 = pca_res[:,0], pca_res[:,1], pca_res[:,2]
        print(f'Explained variation per principal component: {pca.explained_variance_ratio_}')
        df_pca = DataFrame([{'principle component 1': pc1, 'principle component 2': pc2, 'intent': intents[y]} for y,pc1,pc2 in zip(Y,pcs1,pcs2)])
        chart = ggplot(df_pca, aes(x='principle component 1', y='principle component 2', color='factor(intent)')) +\
                    geom_point(alpha=0.8) +\
                    scale_color_discrete() +\
                    ggtitle("First & Second PCs colored by intent")
    
    elif type_ == "tsne":
        tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
        tsne_res = tsne.fit_transform(X)
        tsnes1, tsnes2 = tsne_res[:,0], tsne_res[:,1]
        df_tsne = DataFrame([{'x-tsne': t1, 'y-tsne': t2, 'intent': intents[y]} for y,t1,t2 in zip(Y,tsnes1,tsnes2)])
        chart = ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='factor(intent)')) +\
                    geom_point() +\
                    scale_color_discrete() +\
                    ggtitle("t-SNE dimensions colored by intent")
 
    elif type_ == "pca+tsne":
        n_components = {'webapps': 30, 'chatbot': 50, 'askubuntu': 50, 'chat': 100}[data_source]
        pca = PCA(n_components=n_components)
        pca_res = pca.fit_transform(X)
        print(f'Cumulative explained variation for {n_components} principal components: {np.sum(pca.explained_variance_ratio_)}')
        tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
        tsne_pca_res = tsne.fit_transform(pca_res)
        ts1, ts2 = tsne_pca_res[:,0], tsne_pca_res[:,1]
        df_tsne_pca = DataFrame([{
            'intent': intents[y],
            'x-tsne-pca': t1,
            'y-tsne-pca': t2,
            'label': str(idx)
        } for idx, (y,t1,t2) in enumerate(zip(Y,ts1,ts2))])
        
        means = df_tsne_pca.groupby(['intent'], as_index=False).mean()
        print(means)

        chart = ggplot(df_tsne_pca, aes(x='x-tsne-pca', y='y-tsne-pca', color='factor(intent)', shape='factor(intent)')) +\
                    geom_point(size=8, alpha=0.8) +\
                    geom_point(data=means, size=20) +\
                    scale_color_discrete() +\
                    theme(subplots_adjust={'right': 0.8}) +\
                    labs(
                        title=f'PCA + tSNE embeddings. Dataset: {data_source}, embedding method: {embedding_type}',
                        x = 'x-tsne-pca',
                        y = 'y-tsne-pca'
                    )
                    # geom_label(aes(label='label'), va='bottom') +\

    else:
        print(f"'{type_}'dimensionality reduction is not supported")
        return
    
    if show:
        print(chart)

    return chart
