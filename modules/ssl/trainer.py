from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import ParameterGrid
from torchtext import data
from torchtext.data.example import Example

from modules.models import create_encoder
from modules.train import do_basic_train_and_classify, self_feed
from modules.utilities import repeat_trainer, traina

from modules.utilities.imports import *
from modules.utilities.imports_torch import *

from .eda import eda_corpus
from .encode import encode_data_with_pretrained
from .lp import LabelProp, sigma_fit
from .kmeans import kmeans, recursive_kmeans
from .knn import knn_classify

__all__ = ['repeat_augment_and_train']


def repeat_augment_and_train(dir_to_save, iter_func, model_wrapper, data_source, aug_algo, encoder_model, sim_measure, datasets, text_field, label_field, frac, num_classes, classifier_params, k, learning_type):
    """
    Runs k trials of augmentation & repeat-classification for a given fraction of labeled training data.
    Args:
        dir_to_save (str): directory to save models created/loaded during this process
        aug_algo (str): which augmentation algorithm to use
        encoder_model (str): encoder model to use for augmentation (w similarity measure between these encodings)
        sim_measure (str): which similarity measure to use
        datasets (list(Dataset)): train/val/test torchtext datasets
        text_field (Field): torchtext field for sentences
        label_field (LabelField): torchtext LabelField for class labels
        frac (float): Fraction of labeled training data to use
        classifier_params (dict): params for intent classifier to use on augmented data.
        k (int): Number of times to repeat augmentation-classifier training process
        learning_type (str): inductive|transductive
    Returns:
        8 statistical measures of the results of these trials
    """
    train_ds, val_ds, test_ds = datasets
    class_accs, aug_accs, aug_fracs = [], [], []
    ps, rs, fs = [], [], []

    # FOR ENTROPY HEURISTIC
    # mst_sigmas, entropies, sigmas, accs, fracs = [], [], [], [], []

    # # ABLATION STUDY
    # sigmas, f1_means, f1_stds, aug_acc_means, aug_acc_stds, frac_used_means, frac_used_stds = [],[],[],[],[],[],[]
    # for sigma in np.arange(0.035, 0.155, 0.005):
    #     sigmas.append(sigma)

    for i in tqdm(range(k), total=k):
        examples = train_ds.examples
        np.random.shuffle(examples)
        cutoff = int(frac*len(examples))
        if learning_type == "transductive":
            labeled_examples = train_ds.examples
            unlabeled_examples = test_ds.examples
        elif frac == 0: # 1 labeled eg from each class
            classes_seen = {i: 0 for i in range(num_classes)}
            labeled_examples, unlabeled_examples = [], []
            for eg in examples:
                if classes_seen[eg.y] == 0:
                    labeled_examples.append(eg)
                    classes_seen[eg.y] += 1
                else:
                    unlabeled_examples.append(eg)
        else: # at least one labeled eg from each class
            while True:
                labeled_examples = examples[:cutoff]
                unlabeled_examples = examples[cutoff:]
                if len(set([eg.y for eg in labeled_examples])) == num_classes:
                    break
                np.random.shuffle(examples)

        ##################################################################################################################
        # PROPAGATION PROCESS VISUALISATION (FOR DEMO)
        # from matplotlib import pyplot as plt
        # from pandas import DataFrame
        # from sklearn.decomposition import PCA
        # from sklearn.manifold import TSNE
        # import matplotlib.transforms as transforms

        # # EXTRACT DATA & COMPUTE DIM_REDUCED EMBEDDINGS
        # pickle.dump(labeled_examples, Path(f'./paper/{frac}_labeled_egs.pkl').open('wb'))
        # pickle.dump(unlabeled_examples, Path(f'./paper/{frac}_unlabeled_egs.pkl').open('wb'))
        # labeled_examples = pickle.load(Path(f'./paper/{frac}_labeled_egs.pkl').open('rb'))
        # unlabeled_examples = pickle.load(Path(f'./paper/{frac}_unlabeled_egs.pkl').open('rb'))
        # intents = pickle.load(Path(f'./data/ic/{data_source}/intents.pkl').open('rb'))
        # res = encode_data_with_pretrained(data_source, train_ds, test_ds, text_field, encoder_model, labeled_examples, unlabeled_examples)
        # x_l, y_l, x_u, y_u, _ = res
        # X = np.concatenate([x_l, x_u])
        # Y = np.concatenate([y_l, y_u])
        # pca = PCA(n_components=100)
        # pca_res = pca.fit_transform(X)
        # tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=1000)
        # tsne_pca_res = tsne.fit_transform(pca_res)
        # ts1, ts2 = tsne_pca_res[:,0], tsne_pca_res[:,1]
        # df_tsne_pca = DataFrame([{
        #     'intent': intents[y],
        #     'x-tsne-pca': t1,
        #     'y-tsne-pca': t2,
        #     'og_idx': idx
        # } for idx, (y,t1,t2) in enumerate(zip(Y,ts1,ts2))])
        # df_tsne_pca.to_pickle(f'./paper/{frac}_dataframe.pkl')
        # df_tsne_pca = pd.read_pickle(f'./paper/{frac}_dataframe.pkl')

        # # PLOT INITIAL DATASET
        # fig, ax = plt.subplots()
        # n_l = len(labeled_examples)
        # for idx, intent in enumerate(set(df_tsne_pca['intent'].values)):
        #     values = [v for v in df_tsne_pca.loc[df_tsne_pca['intent']==intent].drop(columns=['intent']).values]
        #     for i, v in enumerate(values):
        #         if v[0] < n_l:
        #             ax.scatter(v[1], v[2], color=f'C{idx}', s=100, alpha=1, label=intent)
        #         else:
        #             ax.scatter(v[1], v[2], color='black', s=100, alpha=0.2)
        # title = 'propagation_initial_labeled_only'
        # for idx, intent in enumerate(set(df_tsne_pca['intent'].values)):
        #     values = [v for v in df_tsne_pca.loc[df_tsne_pca['intent']==intent].drop(columns=['intent']).values]
        #     ax.scatter([v[1] for v in values], [v[2] for v in values], color=f'C{idx}', s=100, alpha=1, label=intent)
        # title = 'propagation_initial_all'
        # ax.grid(b=False)
        # ax.set_ylim(-7.6, 12.5)
        # ax.set_xlim(-10.5, 5.2)
        # fig.set_size_inches(15, 10)
        # plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize='large')
        # plt.savefig(f'./paper/{title}.pdf', format='pdf', dpi=100)
        # plt.show()
        # assert(False)
        
        # # PRELIMINARY DATA FOR MAIN PLOT
        # dim_reduced_points = [0 for _ in range(100)]
        # for idx, intent in enumerate(set(df_tsne_pca['intent'].values)):
        #     values = [v for v in df_tsne_pca.loc[df_tsne_pca['intent']==intent].drop(columns=['intent']).values]
        #     for v in values:
        #         dim_reduced_points[int(v[0])] = (v[1:],intent)
        # data = pickle.load(Path('./paper/propagation_data.pkl').open('rb'))
        # indices = pickle.load(Path('./paper/indices_data.pkl').open('rb'))
        # classifications = pickle.load(Path('./paper/classifications_data.pkl').open('rb'))
        # colors = {'findconnection': 'C1', 'departuretime': 'C0'}
        # intent_map = {0: 'findconnection', 1: 'departuretime'}
        # classified_indices = [0, 1]
        # classified_true_labels = ['findconnection', 'departuretime']
        # classified_intents = ['findconnection', 'departuretime']
        # classified_xs = [dim_reduced_points[i][0][0] for i in classified_indices]
        # classified_ys = [dim_reduced_points[i][0][1] for i in classified_indices]

        # # PLOT EACH RECURSION & PROPAGATION ITERATION
        # with plt.style.context('seaborn-whitegrid'):
        #     plt.rcParams['font.family'] = 'serif'
        #     plt.rcParams['mathtext.fontset'] = 'dejavuserif'

        #     # starting point plot
        #     title = '0_final'
        #     fig, ax = plt.subplots()
        #     unclassified_indices = [i for i in range(100) if i not in classified_indices]
        #     unclassified_xs = [dim_reduced_points[i][0][0] for i in unclassified_indices]
        #     unclassified_ys = [dim_reduced_points[i][0][1] for i in unclassified_indices]
        #     ax.scatter(unclassified_xs, unclassified_ys, color='black', s=100, alpha=0.2)
        #     ax.scatter(classified_xs[1], classified_ys[1], color=colors[classified_intents[1]], marker='s', s=200, alpha=1, label=classified_intents[1])
        #     ax.scatter(classified_xs[0], classified_ys[0], color=colors[classified_intents[0]], marker='s', s=200, alpha=1, label=classified_intents[0])
        #     ax.text(2, 10, 'Recursion 0 -- complete', fontsize=15, color='black', ha="center", va="center")
        #     ax.grid(b=False)
        #     ax.set_ylim(-7.6, 12.5)
        #     ax.set_xlim(-10.5, 5.2)
        #     fig.set_size_inches(15, 10)
        #     plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize='large')
        #     plt.savefig(f'./paper/prop_plots_2/{title}.png', format='png', dpi=150)
        #     plt.close()

        #     for recursion_idx, prop_data in tqdm(enumerate(data), total=len(data)):
        #         # plot results during propagation
        #         Y_us = [prop_data[0]] if len(prop_data) == 1 else np.array(prop_data)[range(0, len(prop_data), 100)]
        #         for prop_idx, Y_u in enumerate(Y_us):
        #             title = f'{recursion_idx+1}_{(prop_idx+1)*100}'
        #             fig, ax = plt.subplots()
        #             for idx, row in enumerate(Y_u):
        #                 color = colors[intent_map[np.argmax(row)]]
        #                 prob = np.max(row)
        #                 ax.scatter(unclassified_xs[idx], unclassified_ys[idx], color=color, s=100, alpha=prob*0.75)
        #             for (x, y, intent, true_label) in zip(classified_xs[2:], classified_ys[2:], classified_intents[2:], classified_true_labels[2:]):
        #                 ax.scatter(x, y, color=colors[intent], marker='s', s=100, alpha=1)
        #                 if intent != true_label:
        #                     ax.scatter(x, y, color='black', marker='x', s=150, alpha=1)
        #             ax.scatter(classified_xs[1], classified_ys[1], color=colors[classified_intents[1]], marker='s', s=200, alpha=1, label=classified_intents[1])
        #             ax.scatter(classified_xs[0], classified_ys[0], color=colors[classified_intents[0]], marker='s', s=200, alpha=1, label=classified_intents[0])
        #             ax.text(2, 10, f'Recursion {recursion_idx+1} -- iterating...', fontsize=15, color='black', ha="center", va="center")
        #             ax.grid(b=False)
        #             ax.set_ylim(-7.6, 12.5)
        #             ax.set_xlim(-10.5, 5.2)
        #             fig.set_size_inches(15, 10)
        #             plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize='large')
        #             plt.savefig(f'./paper/prop_plots_2/{title}.png', format='png', dpi=150)
        #             plt.close()
                
        #         # plot the end result of each recursion - i.e. new ground truth classifications
        #         classified_indices += [i + 2 for i in indices[recursion_idx]]
        #         classified_xs = [dim_reduced_points[i][0][0] for i in classified_indices]
        #         classified_ys = [dim_reduced_points[i][0][1] for i in classified_indices]
        #         classified_true_labels = [dim_reduced_points[i][1] for i in classified_indices]
        #         classified_intents += [intent_map[intent_class] for intent_class in classifications[recursion_idx]]
        #         unclassified_indices = [i for i in range(100) if i not in classified_indices]
        #         unclassified_xs = [dim_reduced_points[i][0][0] for i in unclassified_indices]
        #         unclassified_ys = [dim_reduced_points[i][0][1] for i in unclassified_indices]
        #         title = f'{recursion_idx+1}_final'
        #         fig, ax = plt.subplots()
        #         ax.scatter(unclassified_xs, unclassified_ys, color='black', s=100, alpha=0.2)
        #         for (x, y, intent, true_label) in zip(classified_xs[2:], classified_ys[2:], classified_intents[2:], classified_true_labels[2:]):
        #             ax.scatter(x, y, color=colors[intent], marker='s', s=100, alpha=1)
        #             if intent != true_label:
        #                 ax.scatter(x, y, color='black', marker='x', s=150, alpha=1)
        #         ax.scatter(classified_xs[1], classified_ys[1], color=colors[classified_intents[1]], marker='s', s=200, alpha=1, label=classified_intents[1])
        #         ax.scatter(classified_xs[0], classified_ys[0], color=colors[classified_intents[0]], marker='s', s=200, alpha=1, label=classified_intents[0])
        #         ax.text(2, 10, f'Recursion {recursion_idx+1} -- complete', fontsize=15, color='black', ha="center", va="center")
        #         ax.grid(b=False)
        #         ax.set_ylim(-7.6, 12.5)
        #         ax.set_xlim(-10.5, 5.2)
        #         fig.set_size_inches(15, 10)
        #         plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize='large')
        #         plt.savefig(f'./paper/prop_plots_2/{title}.png', format='png', dpi=150)
        #         plt.close()
    
        # assert(False)
        ##################################################################################################################

        # # ENTROPY HEURISTIC
        # res = encode_data_with_pretrained(data_source, train_ds, test_ds, text_field, encoder_model, labeled_examples, unlabeled_examples)
        # x_l, y_l, x_u, y_u, _ = res
        # mst_sigma, entropy, sigma, acc, frac_used = sigma_fit(x_l, y_l, x_u, y_u, num_classes, data_source)
        # mst_sigmas.append(mst_sigma); entropies.append(entropy); sigmas.append(sigma); accs.append(acc); fracs.append(frac_used)
        # continue

        if aug_algo == "eda":
            x_l, y_l = [eg.x for eg in labeled_examples], [eg.y for eg in labeled_examples]
            augmented_x_l, augmented_y_l = eda_corpus(x_l, y_l)
            new_labeled_data = [{'x': x, 'y': y} for x,y in zip(augmented_x_l, augmented_y_l)]
            augmented_train_examples = [Example.fromdict(x, {'x': ('x', text_field), 'y': ('y', label_field)}) for x in new_labeled_data]
            aug_acc = 1; frac_used = 0
        elif aug_algo == "none":
            augmented_train_examples = labeled_examples
            aug_acc = 1; frac_used = 0
        elif aug_algo == "self_feed":
            sf_thresh = 0.7
            augmented_train_examples, aug_acc, frac_used = self_feed(data_source, dir_to_save, iter_func, model_wrapper, labeled_examples, unlabeled_examples, val_ds, test_ds, text_field, label_field, classifier_params, thresh=sf_thresh)
        else:
            augmented_train_examples, aug_acc, frac_used = augment(data_source, aug_algo, encoder_model, sim_measure, labeled_examples, unlabeled_examples, train_ds, test_ds, text_field, label_field, num_classes, sigma=None)
        
        aug_accs.append(aug_acc); aug_fracs.append(frac_used)
        new_train_ds = data.Dataset(augmented_train_examples, {'x': text_field, 'y': label_field})
        new_datasets = (new_train_ds, val_ds, test_ds)

        if learning_type == "inductive":
            acc, p, r, f = do_basic_train_and_classify(new_train_ds, test_ds, classifier_params, data_source)
        else: # transductive
            predictions = [eg.y for eg in augmented_train_examples[len(train_ds.examples):]]
            test_Y = [eg.y for eg in test_ds.examples]
            acc = accuracy_score(predictions, test_Y)
            avg = "macro avg" if data_source == "chat" else "weighted avg"
            report = classification_report(predictions, test_Y, output_dict=True)[avg]
            p, r, f = report['precision'], report['recall'], report['f1-score']
        
        class_accs.append(acc); ps.append(p); rs.append(r); fs.append(f)

    # # ENTROPY HEURISTIC
    # print(np.mean(entropies), np.std(entropies))
    # print(np.mean(mst_sigmas), np.std(mst_sigmas))
    # print(np.mean(sigmas), np.std(sigmas))
    # print(np.mean(accs), np.std(accs))
    # print(np.mean(fracs), np.std(fracs))
    # assert(False)

    # # ABLATION STUDY
    # print(f"SIGMA: {sigma}")
    # f1_means.append(np.mean(class_accs)); f1_stds.append(np.std(class_accs))
    # aug_acc_means.append(np.mean(aug_accs)); aug_acc_stds.append(np.std(aug_accs))
    # frac_used_means.append(np.mean(aug_fracs)); frac_used_stds.append(np.std(aug_fracs))
    # assert(False)

    print(f"FRAC '{frac}' Results Below:")
    print(f'classification acc --> mean: {np.mean(class_accs)}; std: {np.std(class_accs)}')
    print(f'augmentation acc --> mean: {np.mean(aug_accs)}; std: {np.std(aug_accs)}\t (average frac used: {np.mean(aug_fracs)})')
    print(f'p/r/f1 means --> precision mean: {np.mean(ps)}; recall mean: {np.mean(rs)}; f1 mean: {np.mean(fs)}')
    print(f'p/r/f1 stds --> precision std: {np.std(ps)}; recall std: {np.std(rs)}; f1 std: {np.std(fs)}')

    class_acc_mean, class_acc_std = np.mean(class_accs), np.std(class_accs)
    aug_acc_mean, aug_acc_std, aug_frac_mean = np.mean(aug_accs), np.std(aug_accs), np.mean(aug_fracs)
    p_mean, r_mean, f_mean = np.mean(ps), np.mean(rs), np.mean(fs)
    p_std, r_std, f_std = np.std(ps), np.std(rs), np.std(fs)
    
    # # ABLATION STUDY
    # print([round(s, 3) for s in sigmas])
    # print(f1_means)
    # print(f1_stds)
    # print(aug_acc_means)
    # print(aug_acc_stds)
    # print(frac_used_means)
    # print(frac_used_stds)
    # assert(False)

    return class_acc_mean, class_acc_std, aug_acc_mean, aug_acc_std, aug_frac_mean, p_mean, p_std, r_mean, r_std, f_mean, f_std


def augment(data_source, aug_algo, encoder_model, sim_measure, labeled_examples, unlabeled_examples, train_ds, test_ds, text_field, label_field, num_classes, sigma=None):
    res = encode_data_with_pretrained(data_source, train_ds, test_ds, text_field, encoder_model, labeled_examples, unlabeled_examples)
    x_l, y_l, x_u, y_u, xs_u_unencoded = res

    if aug_algo.startswith("knn"):
        algo_version = aug_algo.split('_')[1]
        if algo_version == 'base':
            classifications, indices = knn_classify(x_l, y_l, x_u, n=1, weights='uniform')
            frac_used = 1
        elif algo_version == 'threshold':
            classifications, indices = knn_classify(x_l, y_l, x_u, n=2, threshold=0.99)
            y_u = y_u[indices]
            xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
            frac_used = float(len(xs_u_unencoded)/len(x_u))
    elif aug_algo.startswith("kmeans"):
        algo_version = aug_algo.split('_')[1]
        if algo_version == "base":
            classifications = kmeans(x_l, x_u, y_l, n_clusters=num_classes)
        elif algo_version == "recursive":
            classifications = recursive_kmeans(x_l, x_u, y_l, n_clusters=num_classes)
        frac_used = 1
    elif aug_algo.startswith("lp"):
        algo_version = aug_algo.split('_')[1]
        lp = LabelProp(x_l, y_l, x_u, y_u, num_classes, data_source=data_source, sigma=sigma)
        if algo_version == 'base':
            lp.propagate()
            classifications, indices = lp.classify(threshold=False)
            y_u = y_u[indices]
            xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
            frac_used = float(len(xs_u_unencoded)/len(x_u))
        elif algo_version == "threshold":
            lp.propagate()
            classifications, indices = lp.classify(threshold=True)
            y_u = y_u[indices]
            xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
            frac_used = float(len(xs_u_unencoded)/len(x_u))
        elif algo_version == "recursive":
            classifications, indices = lp.recursive(x_l, y_l, x_u, y_u)
            y_u = y_u[indices]
            xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
            frac_used = float(len(xs_u_unencoded)/len(x_u))
    
    num_correct = np.sum(classifications == y_u)
    aug_acc = 0 if len(classifications) == 0 else float(num_correct/len(classifications))
    new_labeled_data = [{'x': x, 'y': classifications[i]} for i, x in enumerate(xs_u_unencoded)]
    example_fields = {'x': ('x', text_field), 'y': ('y', label_field)}
    new_examples = [Example.fromdict(x, example_fields) for x in new_labeled_data]

    return labeled_examples + new_examples, aug_acc, frac_used


def train_ic(dir_to_save, iter_func, model_wrapper, text_field, label_field, datasets, classifier_params, return_statistics=False):
    """
    To simplify training (connection to generic training utility with appropriate parameters).
    # ULTIMATELY, classifier_params should just be able to be passed to this train method along with other
    # stuff and THAT method is what sorts it all out... but this is fine for now.
    """
    ps = classifier_params
    wrapper = traina(ps['model_name'], ps['encoder_model'], iter_func, model_wrapper, dir_to_save,
                    nn.CrossEntropyLoss(), datasets, text_field, ps['bs'], ps['encoder_args'],
                    ps['layers'], ps['drops'], ps['lr'], frac=1, verbose=False)
    if return_statistics:
        acc = wrapper.test_accuracy(wrapper.test_di)
        p, r, f, _ = wrapper.test_precision_recall_f1()
        return acc, np.mean(p), np.mean(r), np.mean(f)
    return wrapper


def repeat_ic(dir_to_save, text_field, label_field, datasets, classifier_params, k=10):
    """
    Repeat intent classification training process
    """
    loss_func = nn.CrossEntropyLoss()
    ps = classifier_params
    mean, std, avg_p, avg_r, avg_f = repeat_trainer(ps['model_name'], ps['encoder_model'], get_ic_data_iterators, IntentWrapper, dir_to_save, 
                                    loss_func, datasets, text_field, label_field, ps['bs'], ps['encoder_args'],
                                    ps['layers'], ps['drops'], ps['lr'], frac=1, k=k, verbose=True)
    return mean, std, avg_p, avg_r, avg_f
