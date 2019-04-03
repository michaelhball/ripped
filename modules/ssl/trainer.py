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
from .lp import LabelProp
from .kmeans import kmeans
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
            augmented_train_examples, aug_acc, frac_used = augment(data_source, aug_algo, encoder_model, sim_measure, labeled_examples, unlabeled_examples, train_ds, test_ds, text_field, label_field, num_classes)
        
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

    print(f"FRAC '{frac}' Results Below:")
    print(f'classification acc --> mean: {np.mean(class_accs)}; std: {np.std(class_accs)}')
    print(f'augmentation acc --> mean: {np.mean(aug_accs)}; std: {np.std(aug_accs)}\t (average frac used: {np.mean(aug_fracs)})')
    print(f'p/r/f1 means --> precision mean: {np.mean(ps)}; recall mean: {np.mean(rs)}; f1 mean: {np.mean(fs)}')
    print(f'p/r/f1 stds --> precision std: {np.std(ps)}; recall std: {np.std(rs)}; f1 std: {np.std(fs)}')

    class_acc_mean, class_acc_std = np.mean(class_accs), np.std(class_accs)
    aug_acc_mean, aug_acc_std, aug_frac_mean = np.mean(aug_accs), np.std(aug_accs), np.mean(aug_fracs)
    p_mean, r_mean, f_mean = np.mean(ps), np.mean(rs), np.mean(fs)
    p_std, r_std, f_std = np.std(ps), np.std(rs), np.std(fs)

    return class_acc_mean, class_acc_std, aug_acc_mean, aug_acc_std, aug_frac_mean, p_mean, p_std, r_mean, r_std, f_mean, f_std


def augment(data_source, aug_algo, encoder_model, sim_measure, labeled_examples, unlabeled_examples, train_ds, test_ds, text_field, label_field, num_classes):
    res = encode_data_with_pretrained(data_source, train_ds, test_ds, text_field, encoder_model, labeled_examples, unlabeled_examples)
    xs_l, ys_l, xs_u, ys_u, xs_u_unencoded = res

    if aug_algo.startswith("knn"):
        algo_version = aug_algo.split('_')[1]
        if algo_version == 'base':
            classifications, indices = knn_classify(xs_l, ys_l, xs_u, n=1, weights='uniform')
            frac_used = 1
        elif algo_version == 'threshold':
            classifications, indices = knn_classify(xs_l, ys_l, xs_u, n=2, threshold=0.99)
            ys_u = ys_u[indices]
            xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
            frac_used = float(len(xs_u_unencoded)/len(xs_u))
    elif aug_algo == "kmeans":
        classifications = kmeans(xs_l, xs_u, ys_l, n_clusters=num_classes)
        frac_used = 1
    elif aug_algo.startswith("lp"):
        algo_version = aug_algo.split('_')[1]
        display = True if data_source == "trec" else False
        lp = LabelProp(xs_l, ys_l, xs_u, ys_u, num_classes, data_source=data_source, display=display)
        if algo_version == 'base':
            lp.propagate()
            classifications, indices = lp.classify(threshold=False)
            ys_u = ys_u[indices]
            xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
            frac_used = float(len(xs_u_unencoded)/len(xs_u))
        elif algo_version == "threshold":
            lp.propagate()
            classifications, indices = lp.classify(threshold=True)
            ys_u = ys_u[indices]
            xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
            frac_used = float(len(xs_u_unencoded)/len(xs_u))
        elif algo_version == "recursive":
            classifications, indices = lp.recursive(xs_l, ys_l, xs_u, ys_u)
            ys_u = ys_u[indices]
            xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
            frac_used = float(len(xs_u_unencoded)/len(xs_u))
        elif algo_version == "p1nn":
            classifications, indices = lp.p1nn(xs_l, ys_l, xs_u)
            frac_used = 1
    
    num_correct = np.sum(classifications == ys_u)
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
