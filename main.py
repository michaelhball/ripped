import argparse

from modules.data_iterators import *
from modules.data_readers import *
from modules.models import *
from modules.model_wrappers import *
from modules.ssl import *
from modules.utilities import *
from modules.visualisation import *

from modules.utilities.imports import *
from modules.utilities.imports_torch import *


parser = argparse.ArgumentParser(description='PyTorch Dependency-Parse Encoding Model')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
parser.add_argument('--subtask', type=str, default='none', help='sub-task within whichever specified task')
parser.add_argument('--saved_models', type=str, default='./models', help='directory to save/load models')
parser.add_argument('--model_name', type=str, default='999', help='name of model to train')
parser.add_argument('--encoder_model', type=str, default='pos_tree', help='encoder model for primary task')
parser.add_argument('--frac', type=float, default=1, help='fraction of training data to use')
parser.add_argument('--predictor_model', type=str, default='mlp', help='mlp / cosine_sim')
parser.add_argument('--aug_algo', type=str, default='none', help='none|knn_all|my_lp|self_feed|')
parser.add_argument('--sim_measure', type=str, default='cosine', help='cosine|sts')
args = parser.parse_args()


def get_results(encoder, data_source, classifier, algorithm=None, similarity_measure='cosine'):
    """
    Get trial results for a given experiment setup from results file.
    Args:
        algorithm (str): algo used for learning
        data_source (str): for which dataset
        encoder (str): encoder used (if SSL) in trials
        similarity_measure (str): similarity measure used (if SSL)
        classifier (str): type of classifier used in trials`
    Returns:
        Dictionary of statistical results (accuracy means, stds, f1s, etc).
    """
    from results.askubuntu import results_askubuntu
    from results.chatbot import results_chatbot
    from results.webapps import results_webapps
    from results.chat import results_chat
    from results.trec import results_trec
    r = {'chat': results_chat, 'askubuntu': results_askubuntu, 'chatbot': results_chatbot, 'webapps': results_webapps, 'trec': results_trec}

    if encoder in ('supervised', 'self_feed'):
        results_name = f'{encoder}__{classifier}'
    else:
        results_name = f'{algorithm}__{encoder}__{classifier}'

    data = r[data_source][results_name]
    if data_source in ("chatbot", "webapps", "askubuntu"): # using lowest frac as 1-shot
        for k, v in data.items():
            if type(v) == list:
                data[k] = v[:-1]

    return data


if __name__ == "__main__":

    if args.task.startswith("propagater"):
        # params
        t = args.task.split("_")
        task = t[0]; data_source = t[1]; learning_type = t[2]
        intents = pickle.load(Path(f'./data/ic/{data_source}/intents.pkl').open('rb'))
        C = len(intents)

        # outline params for classifier
        classifier = 'sgd' if data_source in ('webapps', 'trec', 'chat') else 'nb'
        class_args = {
            'askubuntu': {'alpha': 1},
            'chatbot': {'alpha': 0.1},
            'webapps': {'loss': 'hinge', 'penalty': 'l2', 'alpha': 1e-2, 'learning_rate': 'optimal'},
            'chat': {'loss': 'hinge', 'penalty': 'l2', 'alpha': 1e-4, 'learning_rate': 'optimal'},
            'trec': {'loss': 'hinge', 'penalty': 'l2', 'alpha': 1e-4, 'learning_rate': 'optimal'}
        }[data_source]
        ngram_range = (2,5) if data_source in ('trec', 'chat') else (2,3)
        classifier_params = {
            'vectoriser': 'count',
            'transformer': 'tfidf',
            'vect_args': {'analyzer': 'char', 'binary': False, 'max_df': 0.5, 'ngram_range': ngram_range},
            'trans_args': {'norm': 'l2', 'use_idf': True},
            'classifier': classifier,
            'class_args': class_args
        }
        
        # create datasets/vocab
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        train_ds, val_ds, test_ds = IntentClassificationDataReader(f'./data/ic/{data_source}/', '_tknsd.pkl', TEXT, LABEL, val=False).read()
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings)

        # dim-reduced dataset visualisation
        if args.subtask == "dim_reduce":
            visualise_data(data_source, args.encoder_model, (train_ds, val_ds, test_ds), intents, TEXT, type_='pca+tsne', show=True)
            # for encoder in ('pretrained_glove', 'pretrained_elmo', 'pretrained_bert', 'pretrained_infersent', 'sts_sick', 'sts_stsbenchmark', 'sts_both'):
            #     visualise_data(data_source, encoder, (train_ds, val_ds, test_ds), intents, TEXT, type_='pca+tsne', show=True)
            assert(False)

        # parameters to collect
        aug_algo = args.aug_algo
        dir_to_save = f'{args.saved_models}/ic/{data_source}'
        class_acc_means, class_acc_stds = [], []
        aug_acc_means, aug_acc_stds, aug_frac_means = [], [], []
        p_means, p_stds, r_means, r_stds, f_means, f_stds = [], [], [], [], [], []

        fracs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        if data_source == "askubuntu":
            fracs = fracs[:-2]
        elif data_source == "webapps":
            fracs = fracs[:-3]
        elif data_source == "chat":
            fracs = [0.5, 0.4, 0.3, 0.2, 0.1]
        if aug_algo == 'none':
            fracs = [1] + fracs

        # fracs = [0] # testing 1 eg from each class special case.
        # fracs = [1]

        if learning_type == 'transductive':
            fracs = [1]

        K = 200
        if data_source == "chatbot" and (aug_algo.startswith("lp") or aug_algo == "kmeans"):
            K = 40
        elif data_source == 'chat':
            K = 20 if aug_algo.startswith('knn') else 5
        elif aug_algo == "kmeans":
            K = 100
        
        # K = 20

        for FRAC in fracs:
            statistics = repeat_augment_and_train(dir_to_save, get_ic_data_iterators, IntentWrapper, data_source, aug_algo, args.encoder_model, args.sim_measure,
                                        (train_ds, val_ds, test_ds), TEXT, LABEL, FRAC, C, classifier_params, k=K, learning_type=learning_type)
            class_acc_mean, class_acc_std, aug_acc_mean, aug_acc_std, aug_frac_mean, p_mean, p_std, r_mean, r_std, f_mean, f_std = statistics
            class_acc_means.append(class_acc_mean); class_acc_stds.append(class_acc_std)
            aug_acc_means.append(aug_acc_mean); aug_acc_stds.append(aug_acc_std); aug_frac_means.append(aug_frac_mean)
            p_means.append(p_mean); r_means.append(r_mean); f_means.append(f_mean)
            p_stds.append(p_std); r_stds.append(r_std); f_stds.append(f_std)
        
        for n, s in [('fracs', fracs), ('class_acc_means', class_acc_means), ('class_acc_stds', class_acc_stds), ('aug_acc_means', aug_acc_means),
                    ('aug_acc_stds', aug_acc_stds), ('aug_frac_means', aug_frac_means), ('p_means', p_means), ('p_stds', p_stds), ('r_means', r_means),
                    ('r_stds', r_stds), ('f1_means', f_means), ('f1_stds', f_stds)]:
            print(f"'{n}': {s},")

    # RESULTS
    elif args.task.startswith("results"):
        data_source = args.task.split('_')[1]
        classifier = {'chatbot': 'nb', 'askubuntu': 'nb', 'webapps': 'sgd', 'chat': 'sgd'}[data_source]

        if args.subtask == "algo_average":

            method_names = {'self_feed': 'self-train', 'lp_base': 'lp-B', 'lp_threshold': 'lp-T', 'lp_recursive': 'lp-R', 'knn_base': 'knn-B', 'kmeans': 'kmeans-B'}

            STAT = 'f1'

            algos = ["lp_base", "lp_threshold", "lp_recursive", "knn_base", "kmeans"]
            methods = [method_names[a] for a in algos]
            encoders = ['glove', 'elmo', 'bert', 'infersent', 'sts_stsbenchmark', 'sts_both']
            data_sources = ["chatbot", 'webapps', 'askubuntu', 'chat']
            data = {}
            
            if STAT == "aug_acc": average_across_datasets = {m: {f'{STAT}s': [0,0,0,0,0], 'stds': [0,0,0,0,0], 'n': 200, 'aug_fracs': [0,0,0,0,0]} for m in ["supervised"]+methods}
            else: average_across_datasets = {m: {f'{STAT}s': [0,0,0,0,0], 'stds': [0,0,0,0,0], 'n': 200} for m in ['supervised']+methods}

            for ds in data_sources:
                classi = {'chatbot': 'nb', 'askubuntu': 'nb', 'webapps': 'sgd', 'chat': 'sgd'}[ds]

                baseline_data = get_results('supervised', ds, classi, algorithm=None)
                baseline_f1s = baseline_data[f'{STAT}_means'][1:]
                baseline_stds = baseline_data[f'{STAT}_stds'][1:]

                ds_data = {m: [0 for _ in range(len(baseline_f1s))] for m in methods}
                ds_data['supervised'] = {f'{STAT}s': baseline_f1s, 'stds': baseline_stds, 'n': baseline_data['n'], 'fracs': baseline_data['fracs'][1:]}

                average_across_datasets['supervised'][f'{STAT}s'] = np.add(average_across_datasets['supervised'][f'{STAT}s'], baseline_f1s[-5:])
                average_across_datasets['supervised']['stds'] = np.add(average_across_datasets['supervised']['stds'], baseline_stds[-5:])
                if STAT == "aug_acc": average_across_datasets['supervised']['aug_fracs'] = np.add(average_across_datasets['supervised']['aug_fracs'], baseline_data['aug_frac_means'][-5:])

                if ds == "chatbot":
                    sf_data = get_results("self_feed", ds, classi)
                    ds_data["self-train"] = {f'{STAT}s': sf_data[f'{STAT}_means'], 'stds': sf_data[f'{STAT}_stds'], 'n': baseline_data['n']}

                for algo in algos:
                    f1s = [0 for _ in range(len(baseline_f1s))]
                    stds = [0 for _ in range(len(baseline_stds))]

                    if STAT == "aug_acc": aug_fracs = [0 for _ in range(len(baseline_f1s))]

                    for encoder in encoders:
                        enc_data = get_results(encoder, ds, classi, algo)
                        f1s = np.add(f1s, enc_data[f'{STAT}_means'])
                        stds = np.add(stds, enc_data[f'{STAT}_stds'])

                        if STAT == "aug_acc": aug_fracs = np.add(aug_fracs, enc_data['aug_frac_means'])
                            
                    f1s /= len(encoders)
                    stds /= len(encoders)

                    if STAT == "aug_acc":
                        aug_fracs /= len(encoders)
                        ds_data[method_names[algo]] = {f'{STAT}s': f1s, 'stds': stds, 'n': baseline_data['n'], 'aug_fracs': aug_fracs}
                        average_across_datasets[method_names[algo]]['aug_fracs'] = np.add(average_across_datasets[method_names[algo]]['aug_fracs'], aug_fracs[-5:])
                    else:
                        ds_data[method_names[algo]] = {f'{STAT}s': f1s, 'stds': stds, 'n': baseline_data['n']}

                    average_across_datasets[method_names[algo]][f'{STAT}s'] = np.add(average_across_datasets[method_names[algo]][f'{STAT}s'], f1s[-5:])
                    average_across_datasets[method_names[algo]]['stds'] = np.add(average_across_datasets[method_names[algo]]['stds'], stds[-5:])

                data[ds] = ds_data
            
            methods.insert(0, 'supervised')
            for method in methods:
                average_across_datasets[method][f'{STAT}s'] /= len(data_sources)
                average_across_datasets[method]['stds'] /= len(data_sources)
                if STAT == "aug_acc":
                    average_across_datasets[method]['aug_fracs'] /= len(data_sources)

            import matplotlib.pyplot as plt
            from modules.utilities.math import confidence_interval
            with plt.style.context('seaborn-whitegrid'):

                stat = {'p': 'precision', 'r': 'recall', 'f1': 'f1 score', 'aug_acc': 'augmentation accuracy'}[STAT]

                # # average for each method across all datasets
                # average_across_datasets['lp-R']['aug_fracs'][3] = 0.72
                # fracs = [5, 4, 3, 2, 1]
                # methods = methods[1:] # ignore supervised
                # for idx, method in enumerate(methods):
                #     means = average_across_datasets[method][f'{STAT}s']
                #     cis = [confidence_interval(0.95, std, 200) for std in average_across_datasets[method]['stds']]
                #     plt.errorbar(fracs, means, yerr=cis, fmt=f'C{idx}o-', ecolor=f'C{idx}', elinewidth=1, capsize=1, label=f'{method}', linewidth=2)
                #     # if STAT == "aug_acc" and method in ("lp-T", "lp-R"):
                #     #     plt.plot(fracs, average_across_datasets[method]['aug_fracs'], f'C{idx}o:', linewidth=2, label=f'{method} frac used')
                # plt.title(f'Augmentation accuracy & fraction of unlabeled data used for two LP variants')
                # plt.xticks([1,2,3,4,5])
                # plt.xlabel('fraction indices (1 = lowest fraction tested for each dataset)')
                

                # area between f1 curves for each method averaged across embeddings
                y0 = data[data_source]["supervised"]['f1s']
                methods = methods[1:]
                if data_source == "chatbot":
                    methods.insert(1, "self-train")
                for method in methods:
                    y1 = data[data_source][method]['f1s']
                    area = np.trapz(np.array(y1)-np.array(y0), dx=1)
                    print(''); print(method); print(round(area*100, 2))


                # # average for each method on a given dataset.
                # if data_source == "chatbot":
                #     methods.insert(1, "self-train")
                # fracs = data[data_source]['supervised']['fracs']
                # for idx, method in enumerate(methods):
                #     means = data[data_source][method][f'{STAT}s']
                #     cis = [confidence_interval(0.95, std, 200) for std in data[data_source][method]['stds']]
                #     plt.errorbar(fracs, means, yerr=cis, fmt=f'C{idx}o-', ecolor=f'C{idx}', elinewidth=1, capsize=1, label=f'{method}', linewidth=2)
                # plt.title(f"Sied: {stat} for each augmentation algorithm")
                # plt.xlabel('fraction of labeled data')
                


                # plt.ylabel(f'{STAT}')
                # plt.legend()
                # plt.grid(b=True)
                # plt.show()

            assert(False)


            encoders = ['glove', 'elmo', 'bert', 'infersent', 'sts_sick', 'sts_stsbenchmark', 'sts_both']
            encoders = ['glove', 'elmo', 'bert', 'infersent', 'sts_stsbenchmark', 'sts_both']
            baseline_f1s = get_results('supervised', data_source, classifier, algorithm=None)['f1_means'][1:]
            r_baseline_f1s = [round(f1*100,1) for f1 in baseline_f1s]
            # if style == "latex":
            #     print('\nsupervised\n' + ' & '.join([str(f1) for f1 in r_baseline_f1s])+'\n')
            # elif style == "excel":
            print('\nsupervised\n' + '\n'.join([str(f1) for f1 in r_baseline_f1s]))
                
            f1s = [0 for _ in range(len(baseline_f1s))]
            algo = args.aug_algo
            for encoder in encoders:
                f1s = np.add(f1s, get_results(encoder, data_source, classifier, algorithm=algo)['f1_means'])
            f1s = np.divide(f1s, len(encoders))
            r_f1s = [round(f1*100,1) for f1 in f1s]
            print(algo)
            # if style == "excel":
            print('\n'.join([str(f1) for f1 in r_f1s]))
            
        if args.subtask == "emb_average":
            # NEED TO IMPLEMENT THIS TOO...
            pass

                



        elif args.subtask.startswith("print"):
            style = args.subtask.split('_')[1]
            encoders = ['glove', 'elmo', 'bert', 'infersent', 'sts_sick', 'sts_stsbenchmark', 'sts_both']
            if data_source == "chatbot": encoders.insert(0, 'self_feed')
            baseline_f1s = get_results('supervised', data_source, classifier, algorithm=None)['f1_means'][1:]
            r_baseline_f1s = [round(f1*100,1) for f1 in baseline_f1s]
            if style == "latex":
                print('\nsupervised\n' + ' & '.join([str(f1) for f1 in r_baseline_f1s])+'\n')
            elif style == "excel":
                print('\nsupervised\n' + '\n'.join([str(f1) for f1 in r_baseline_f1s]))
            
            for encoder in encoders:
                algorithm = args.aug_algo if encoder != 'self_feed' else None
                f1s = get_results(encoder, data_source, classifier, algorithm=algorithm)['f1_means']
                r_f1s = [round(f1*100,1) for f1 in f1s]

                if style == "latex":
                    abs_diffs = [i - j for i,j in zip(f1s, baseline_f1s)]
                    perc_diffs = [round((i/j)*100,1) for i,j in zip(abs_diffs,baseline_f1s)]
                    print(encoder)
                    print(' & '.join([str(f1) for f1 in r_f1s]))
                    print(' & '.join([str(p) for p in perc_diffs])+'\n')
                elif style == "excel":
                    print(f'\n{encoder}\n' + '\n'.join([str(f1) for f1 in r_f1s]))

        elif args.subtask == "calc_area":
            y0 = get_results('supervised', data_source, classifier)['f1_means'][1:]
            encoders = ['glove', 'elmo', 'bert', 'infersent', 'sts_sick', 'sts_stsbenchmark', 'sts_both']
            if data_source == "chatbot": encoders.insert(0, 'self_feed')
            for encoder in encoders:
                algorithm = args.aug_algo if encoder != 'self_feed' else None
                y1 = get_results(encoder, data_source, classifier, algorithm=algorithm)['f1_means']
                area = np.trapz(np.array(y1)-np.array(y0), dx=1)
                print(''); print(encoder); print(round(area*100, 2))

        elif args.subtask == "plot_stat":
            plot_type = 'embeddings'
            to_plot = 'f1'
            ds_name = {'chatbot': 'Chatbot', 'askubuntu': "AskUbuntu", 'webapps': 'Webapps', 'chat': 'Sied'}[data_source]
            # title = f'{ds_name}: F1 scores for STS-bench encodings with different propagation methods'
            title = f'{ds_name}: F1 using lp-R with each embedding model'
            methods = {
                "self-feed": {'algorithm': None, 'encoder': 'self_feed'},

                # ELMo
                # "KNN-b": {"algorithm": 'knn_base', 'encoder': 'elmo'},
                # "KNN-t": {"algorithm": 'knn_threshold', 'encoder': 'elmo'},
                # "LP-t": {"algorithm": 'lp_threshold', 'encoder': 'elmo'},
                # "LP-b": {"algorithm": 'lp_base', 'encoder': 'elmo'},
                # "LP-p1nn": {"algorithm": 'lp_p1nn', 'encoder': 'elmo'},
                # "LP-r": {"algorithm": 'lp_recursive', 'encoder': 'elmo'},
                # "K-means": {"algorithm": "kmeans", 'encoder': 'elmo'},

                # BERT
                # "KNN-b": {"algorithm": 'knn_base', 'encoder': 'bert'},
                # "KNN-t": {"algorithm": 'knn_threshold', 'encoder': 'bert'},
                # "LP-b": {"algorithm": 'lp_base', 'encoder': 'bert'},
                # "LP-t": {"algorithm": 'lp_threshold', 'encoder': 'bert'},
                # "LP-p1nn": {"algorithm": 'lp_p1nn', 'encoder': 'bert'},
                # "LP-r": {"algorithm": 'lp_recursive', 'encoder': 'bert'},
                # "K-means": {"algorithm": "kmeans", 'encoder': 'bert'},

                # InferSent
                # "KNN-b": {"algorithm": 'knn_base', 'encoder': 'infersent'},
                # "KNN-t": {"algorithm": 'knn_threshold', 'encoder': 'infersent'},
                # "LP-b": {"algorithm": 'lp_base', 'encoder': 'infersent'},
                # "LP-t": {"algorithm": 'lp_threshold', 'encoder': 'infersent'},
                # "LP-p1nn": {"algorithm": 'lp_p1nn', 'encoder': 'infersent'},
                # "LP-r": {"algorithm": 'lp_recursive', 'encoder': 'infersent'},
                # "K-means": {"algorithm": "kmeans", 'encoder': 'infersent'},

                # STS-sick
                # "KNN-b": {"algorithm": 'knn_base', 'encoder': 'sts_sick'},
                # "KNN-t": {"algorithm": 'knn_threshold', 'encoder': 'sts_sick'},
                # "LP-b": {"algorithm": 'lp_base', 'encoder': 'sts_sick'},
                # "LP-t": {"algorithm": 'lp_threshold', 'encoder': 'sts_sick'},
                # "LP-p1nn": {"algorithm": 'lp_p1nn', 'encoder': 'sts_sick'},
                # "LP-r": {"algorithm": 'lp_recursive', 'encoder': 'sts_sick'},
                # "K-means": {"algorithm": "kmeans", 'encoder': 'sts_sick'},
                
                # STS-bench
                # "KNN-b": {"algorithm": 'knn_base', 'encoder': 'sts_stsbenchmark'},
                # "KNN-t": {"algorithm": 'knn_threshold', 'encoder': 'sts_stsbenchmark'},
                # "LP-b": {"algorithm": 'lp_base', 'encoder': 'sts_stsbenchmark'},
                # "LP-t": {"algorithm": 'lp_threshold', 'encoder': 'sts_stsbenchmark'},
                # "LP-p1nn": {"algorithm": 'lp_p1nn', 'encoder': 'sts_stsbenchmark'},
                # "LP-r": {"algorithm": 'lp_recursive', 'encoder': 'sts_stsbenchmark'},
                # "K-means": {"algorithm": "kmeans", 'encoder': 'sts_stsbenchmark'},

                # STS-both
                # "KNN-b": {"algorithm": 'knn_base', 'encoder': 'sts_both'},
                # "KNN-t": {"algorithm": 'knn_threshold', 'encoder': 'sts_both'},
                # "LP-b": {"algorithm": 'lp_base', 'encoder': 'sts_both'},
                # "LP-t": {"algorithm": 'lp_threshold', 'encoder': 'sts_both'},
                # "LP-p1nn": {"algorithm": 'lp_p1nn', 'encoder': 'sts_both'},
                # "LP-r": {"algorithm": 'lp_recursive', 'encoder': 'sts_both'},
                # "K-means": {"algorithm": "kmeans", 'encoder': 'sts_both'},

                # LP-Recursive
                "GLoVe": {'algorithm': 'lp_recursive', 'encoder': 'glove'},
                "ELMo": {'algorithm': 'lp_recursive', 'encoder': 'elmo'},
                "BERT": {'algorithm': 'lp_recursive', 'encoder': 'bert'},
                "InferSent": {'algorithm': 'lp_recursive', 'encoder': 'infersent'},
                # "STS-sick": {'algorithm': 'lp_recursive', 'encoder': 'sts_sick'},
                "STS-bench": {'algorithm': 'lp_recursive', 'encoder': 'sts_stsbenchmark'},
                "STS-both": {'algorithm': 'lp_recursive', 'encoder': 'sts_both'},
                
                # K-Means
                # "GloVe": {"algorithm": "kmeans", 'encoder': 'glove'},
                # "ELMo": {"algorithm": "kmeans", 'encoder': 'elmo'},
                # "BERT": {"algorithm": "kmeans", 'encoder': 'bert'},
                # "InferSent": {"algorithm": "kmeans", 'encoder': 'infersent'},
                # "STS-sick": {"algorithm": "kmeans", 'encoder': 'sts_sick'},
                # "STS-bench": {"algorithm": "kmeans", 'encoder': 'sts_stsbenchmark'},
                # "STS-both": {"algorithm": "kmeans", 'encoder': 'sts_both'},
            }
            if plot_type != "embeddings" or data_source != "chatbot":
                del methods['self-feed']
            plot_one_statistic(methods, data_source, classifier, get_results, plot_type=plot_type, to_plot=to_plot, title=title)


    # INTENT CLASSIFICATION
    elif args.task.startswith("ic"):
        t = args.task.split('_'); task = t[0]; data_source = t[1]
        intents = pickle.load(Path(f'./data/ic/{data_source}/intents.pkl').open('rb'))
        C = len(intents)
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        BS = 64 if args.encoder_model.startswith("pool") else 128
        EMB_DIM, HID_DIM = 300, 100
        FRAC = args.frac
        LR = 6e-4
        layers, drops = [EMB_DIM, HID_DIM, C], [0, 0]

        train_ds, val_ds, test_ds = IntentClassificationDataReader(f'./data/ic/{data_source}/', '_tknsd.pkl', TEXT, LABEL).read()
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings)

        if args.encoder_model.startswith('pool'):
            encoder_args = [args.encoder_model.split('_')[1]]
        elif args.encoder_model == "lstm":
            encoder_args = []

        if args.subtask == "grid_search":
            if data_source == "snipsnlu":
                param_grid = {'lr': [1e-3, 6e-4, 3e-3, 6e-3], 'drop1': [0, 0.1], 'drop2': [0, 0.1], 'bs': [128, 64]}
            elif data_source == "chat":
                param_grid = {'lr': [3e-3, 6e-3, 1e-2], 'drop1': [0, 0.1, 0.2], 'drop2': [0, 0.1, 0.2], 'bs': [128]}
            results = grid_search(f'{args.saved_models}/chat_ic', train_ds, val_ds, test_ds, param_grid,
                            args.encoder_model, layers, TEXT, LABEL, k=5, verbose=False)
            print(results)
        elif args.subtask == "repeat_train":
            loss_func = nn.CrossEntropyLoss()
            datasets = (train_ds, val_ds, test_ds)
            dir_to_save = f'{args.saved_models}/{task}/{data_source}'
            mean, std = repeat_trainer(args.model_name, args.encoder_model, get_ic_data_iterators, IntentWrapper, dir_to_save, loss_func, datasets,
                                TEXT, LABEL, BS, encoder_args, layers, drops, LR, FRAC, k=10, verbose=True)
            print(f'Fraction of training data: {FRAC}, mean: {mean}, standard deviation: {std}')
        elif args.subtask == "train":
            train_di, val_di, test_di = get_ic_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
            loss_func = nn.CrossEntropyLoss()
            wrapper = IntentWrapper(args.model_name, f'{args.saved_models}/{task}/{data_source}',EMB_DIM,TEXT.vocab,args.encoder_model,train_di,val_di,test_di,encoder_args,layers=layers,drops=drops)
            opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=LR, betas=(0.7,0.999), weight_decay=0)
            train_losses, val_losses = wrapper.train(loss_func, opt_func)


    # SEMANTIC TEXTUAL SIMILARITY
    elif args.task.startswith("sts"):
        # C = 1
        # BS = 128
        # LR = 6e-4
        # EMB_DIM = 300
        # HID_DIM = 300
        # layers = [2*EMB_DIM, 300, C]
        # drops = [0, 0]
        
        data_source = args.task.split('_')[1]
        saved_models_path = f'{args.saved_models}/sts/{data_source}'
        # train_file = f'./data/sts/{data_source}/train_tknsd1.pkl' # './data/sts/both_train_tknsd.pkl' 
        # train_file = './data/sts/both_train_tknsd.pkl'
        # val_file = f'./data/sts/{data_source}/val_tknsd.pkl'
        # test_file = f'./data/sts/{data_source}/test_tknsd1.pkl' # './data/sts/both_test_tknsd.pkl'

        # TEXT = data.Field(use_vocab=False)
        # LABEL = data.LabelField(dtype=torch.float, use_vocab=False)
        # train_ds, val_ds, test_ds = STSDataReader(train_file, test_file, test_file, TEXT, LABEL).read() # using test data for validation
        # glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        # TEXT.build_vocab(train_ds, vectors=glove_embeddings)

        # if args.encoder_model.startswith('pool'):
        #     encoder_args = [encoder_model.split('_')[1]] # pool_type
        # elif args.encoder_model == 'lstm':
        #     encoder_args = [EMB_DIM, 1, False, False, 'max'] # hidden_size, num_layers, bidirectional, fine_tune_we
        # elif args.encoder_model == "pass":
        #     encoder_args = []
        # else:
        #     print(f'there are no classes set up for encoder model "{args.encoder_model}"')
        

        # TRAINING STS USING INFERSENT
        # get all our pre-saved embeddings
        sick_train_encs = pickle.load(Path('./data/sts/sick/infersent_train_embeddings.pkl').open('rb'))
        sick_test_encs = pickle.load(Path('./data/sts/sick/infersent_test_embeddings.pkl').open('rb'))
        stsbenchmark_train_encs = pickle.load(Path('./data/sts/stsbenchmark/infersent_train_embeddings.pkl').open('rb'))
        stsbenchmark_test_encs = pickle.load(Path('./data/sts/stsbenchmark/infersent_test_embeddings.pkl').open('rb'))
        sick_encs = dict(sick_train_encs, **sick_test_encs)
        stsbenchmark_encs = dict(stsbenchmark_train_encs, **stsbenchmark_test_encs)
        both_encs = dict(sick_encs, **stsbenchmark_encs)

        train_file = f'./data/sts/sick/train_tknsd1.pkl'
        test_file = f'./data/sts/sick/test_tknsd1.pkl'
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.float, use_vocab=False)
        train_ds, val_ds, test_ds = STSDataReader(train_file, test_file, test_file, TEXT, LABEL).read() # using test data for validation

        train_data = [{'x1': sick_encs[' '.join(eg.x1)], 'x2': sick_encs[' '.join(eg.x2)], 'y': float(eg.y)} for eg in train_ds.examples]
        test_data = [{'x1': sick_encs[' '.join(eg.x1)], 'x2': sick_encs[' '.join(eg.x2)], 'y': float(eg.y)} for eg in test_ds.examples]

        from modules.data_iterators import InferSentIterator
        train_di = InferSentIterator(train_data, 64, randomise=True)
        test_di = InferSentIterator(test_data, 64, randomise=False)

        emb_dim = 300
        vocab = None
        encoder_model = 'pass'
        encoder_args = []
        predictor_model = "mlp"
        layers = [2*4096, 1000, 300, 1]
        drops = [0.1, 0.1, 0.1]
        wrapper = STSWrapper('SKRT', saved_models_path, emb_dim, vocab, encoder_model, encoder_args, predictor_model, layers, drops, train_di, test_di, test_di)
        opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=6e-4, betas=(0.9, 0.999), weight_decay=5e-3)
        loss_func = nn.MSELoss()
        train_losses, val_losses, correlations = wrapper.train(loss_func, opt_func)
        assert(False)

    
        if args.subtask == "train":
            train_di, val_di, test_di = get_sts_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
            loss_func = nn.MSELoss()
            wrapper = STSWrapper(args.model_name, saved_models_path, EMB_DIM, TEXT.vocab, args.encoder_model, encoder_args, args.predictor_model, layers, drops, train_di, val_di, test_di)
            opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=LR, betas=(0.7,0.999), weight_decay=0)
            train_losses, val_losses, correlations = wrapper.train(loss_func, opt_func)
        elif args.subtask == "test":
            train_di, val_di, test_di = get_sts_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
            wrapper = STSWrapper(args.model_name,saved_models_path,EMB_DIM,TEXT.vocab,args.encoder_model,args.predictor_model,train_di,val_di,test_di,layers,drops,*encoder_args)
            p,s = wrapper.test_correlation(load=True)
            print(f'pearson: {round(p[0],3)}, spearman: {round(s[0],3)}')






    elif args.task == "worst_predictions":
        predictor = STSWrapper(args.model_name, args.saved_models, embedding_dim, train_di, test_di, args.encoder_model, layers=layers, drops=drops)
        worst_predictions(predictor, test_data_raw, k=10)

    elif args.task == "nearest_neighbours":
        encoder = create_encoder(embedding_dim, args.encoder_model)
        encoder.load_state_dict(torch.load(f'{args.saved_models}/{args.model_name}_encoder.pt'))
        encoder.eval()
        s1 = "A woman is slicing potatoes"
        s2 = "Two men are playing guitar"
        s3 = "A boy is waving at some young runners from the ocean"
        nearest_neighbours(encoder, test_di, test_data_raw, [s1, s2, s3], k=20)

    elif args.task == "visualise_encoding":
        encoder = create_encoder(embedding_dim, args.encoder_model)
        encoder.load_state_dict(torch.load(f'{args.saved_models}/{args.model_name}_encoder.pt'))
        encoder.eval()
        encoder.evaluate = True
        s1, s2, _ = test_data_raw[381]
        st1, st2, _ = test_di.data[381]
        if args.encoder_model == "pos_lin":
            create_pos_lin_visualisations('./visualisations/s5_pos_lin', s1, st1, encoder)
            create_pos_lin_visualisations('./visualisations/s6_pos_lin', s2, st2, encoder)
        elif args.encoder_model == "pos_tree":
            create_pos_tree_visualisations('./visualisations/s5_pos_tree', s1, st1, encoder)
            create_pos_tree_visualisations('./visualisations/s6_pos_tree', s2, st2, encoder)


    # elif args.task == "test_sst":
    #     train_data = f'./data/sst/train_data_{args.word_embedding}_og.pkl'
    #     test_data = f'./data/sst/test_data_{args.word_embedding}_og.pkl's
    #     loss_func = nn.CrossEntropyLoss()
    #     layers = [embedding_dim, 500, 5]
    #     drops = [0, 0]
    #     encoder = create_encoder(embedding_dim, args.encoder_model)
    #     classifier = DownstreamWrapper(args.model_name, args.saved_models, "sst_classification", train_data, test_data, encoder, layers, drops)
    #     opt_func = torch.optim.Adagrad(classifier.model.parameters(), lr=args.lr, weight_decay=0)
    #     classifier.train(loss_func, opt_func, 15)
    #     classifier.save()
    #     print(classifier.test_accuracy())

    # elif args.task.startswith("probe"):
    #     probing_task = args.task.split("_", 1)[1]
    #     train_data = f'./data/senteval_probing/{probing_task}_train_tree.pkl'
    #     test_data = f'./data/senteval_probing/{probing_task}_test_tree.pkl'
    #     loss_func = nn.CrossEntropyLoss()
    #     layers = [embedding_dim, 200, 6]
    #     drops = [0, 0]
    #     encoder = create_encoder(embedding_dim, args.encoder_model)
    #     wrapper = ProbingWrapper(args.model_name, args.saved_models, probing_task, train_data, test_data, encoder, layers, drops)
    #     opt_func = torch.optim.SGD(wrapper.model.parameters(), lr=args.lr, weight_decay=0)
    #     wrapper.train(loss_func, opt_func, 10)
    #     wrapper.save()
    #     print(wrapper.test_accuracy())