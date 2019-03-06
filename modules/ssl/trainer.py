from flair.data import Sentence
from flair.embeddings import BertEmbeddings, ELMoEmbeddings, FlairEmbeddings, StackedEmbeddings
from sklearn.model_selection import ParameterGrid
from torchtext import data
from torchtext.data.example import Example

from modules.data_iterators import get_ic_data_iterators
from modules.models import create_encoder
from modules.model_wrappers import IntentWrapper
from modules.utilities import repeat_trainer, traina

from modules.utilities.imports import *
from modules.utilities.imports_torch import *

from .label_propagation import label_prop_classify, LabelProp
from .knn import knn_classify


__all__ = ['repeat_augment_and_train', 'grid_search_lp', 'self_train']


def grid_search_lp(data_source, lp_type, encoder_model, train_ds, text_field, label_field, frac):
    if lp_type == "propagation":
        pg = {'kernel': ['knn'], 'gamma': [100], 'n_neighbors': [1], 'alpha': [None], 'max_iter': [1000000], 'tol': [1e-3]} # BEST param set if knn
        # pg = {'kernel': ['rbf'], 'gamma': [20, 10, 30, 50], 'n_neighbors': [1], 'alpha': [None], 'max_iter': [500, 1000, 2000, 10000, 100000], 'tol': [1e-3, 1e-4]}
    else:
        # only need to find the best alpha, but can do this using labeled + unlabeled addition below...
        pg = {'kernel': ['knn'], 'gamma': [20], 'n_neighbors': [1], 'alpha': [0.1,0.2,0.3,0.4,0.5], 'max_iter': [30], 'tol': [1e-3]}

    results = []
    param_grid = ParameterGrid(pg)
    for i, params in tqdm(enumerate(param_grid), total=len(param_grid)):
        pd = {'kernel': params['kernel'], 'gamma': params['gamma'], 'n_neighbors': params['n_neighbors'],
                'alpha': params['alpha'], 'max_iter': params['max_iter'], 'tol': params['tol']}

        accs = []
        for idx in tqdm(range(10), total=10):
            examples = train_ds.examples
            np.random.shuffle(examples)
            cutoff = int(frac*len(examples))
            labeled_examples = examples[:cutoff]
            unlabeled_examples = examples[cutoff:]

            embedding_type = encoder_model.split('_')[1]
            res = encode_data_with_pretrained(data_source, train_ds, text_field, embedding_type, labeled_examples, unlabeled_examples)
            xs_l, ys_l, xs_u, ys_u, xs_u_unencoded = res
            # should append xs_u to xs_l, and append labels of -1 for each of these samples (this will be the input to the fit function).
            # definitely need to try this too.

            xs_l = np.array([x / np.linalg.norm(x) for x in xs_l])
            xs_u = np.array([x / np.linalg.norm(x) for x in xs_u])

            classifications = label_prop_classify(xs_l, ys_l, xs_u, lp_type, pd)
            accs.append(np.sum(classifications == ys_u) / len(ys_u))
        
        mean, std = np.mean(accs), np.std(accs)
        # print("-------------------------------------------------------------------------------------------------------------------------------------------")
        # print(i, mean, std, params)
        # print("-------------------------------------------------------------------------------------------------------------------------------------------")
        results.append({'mean': mean, 'std': std, 'params': params})
    
    return list(reversed(sorted(results, key=lambda x: x['mean'])))


def self_feed(dir_to_save, labeled_examples, unlabeled_examples, val_ds, test_ds, text_field, label_field, classifier_params):
    """
    Implements a self-training algorithm.
    Args:
        dir_to_save (str): directory to save models created/loaded during this process
        labeled_examples (list(Example)): 
        unlabeled_examples (list(Example)): 
        text_field (Field): torchtext field for sentences
        label_field (LabelField): torchtext LabelField for class labels
        classifier_params (dict): all params needed to instantiate an intent classifier
    Returns:
        augmented labeled examples, accuracy of augmentation, fraction of unlabeled data used in augmentation
    """
    initial_labeled_num = len(labeled_examples)
    initial_unlabeled_num = len(unlabeled_examples)
    num_correct, num_added = 0, 0
    
    while True:
        new_train_ds = data.Dataset(labeled_examples, {'x': text_field, 'y': label_field})
        wrapper = train_ic(dir_to_save, text_field, label_field, (new_train_ds, val_ds, test_ds), classifier_params)
        confidences, preds = wrapper.classify_all(unlabeled_examples)

        new_unlabeled_examples = []
        for i, (conf, pred) in enumerate(zip(confidences, preds)):
            if conf > 0.98: # PLAY AROUND WITH THIS THRESHOLD
                labeled_examples.append(Example.fromdict({'x': unlabeled_examples[i].x, 'y': pred}, {'x': ('x', text_field), 'y': ('y', label_field)}))
                num_added += 1
                if pred == unlabeled_examples[i].y:
                    num_correct += 1
            else:
                new_unlabeled_examples.append(unlabeled_examples[i])
        
        if len(unlabeled_examples) - len(new_unlabeled_examples) == 0:
            break
        unlabeled_examples = new_unlabeled_examples
        np.random.shuffle(labeled_examples)

    return labeled_examples, float(num_correct/num_added), float(num_added/initial_unlabeled_num)


def repeat_augment_and_train(dir_to_save, data_source, aug_algo, encoder_model, datasets, text_field, label_field, frac, classifier_params, k=5):
    """
    Runs k trials of augmentation & repeat-classification for a given fraction of labeled training data.
    Args:
        dir_to_save (str): directory to save models created/loaded during this process
        aug_algo (str): which augmentation algorithm to use
        encoder_model (str): encoder model to use for augmentation (w similarity measure between these encodings)
        datasets (list(Dataset)): train/val/test torchtext datasets
        text_field (Field): torchtext field for sentences
        label_field (LabelField): torchtext LabelField for class labels
        frac (float): Fraction of labeled training data to use
        classifier_params (dict): params for intent classifier to use on augmented data.
        k (int): Number of times to repeat augmentation-classifier training process
    Returns:
        8 statistical measures of the results of these trialss
    """
    train_ds, val_ds, test_ds = datasets
    aug_accs, aug_fracs = [], []
    means, stds = [], []
    ps, rs, fs = [], [], []
    for i in range(k):
        examples = train_ds.examples
        np.random.shuffle(examples)
        cutoff = int(frac*len(examples))
        labeled_examples = examples[:cutoff]
        unlabeled_examples = examples[cutoff:]

        if aug_algo == "self_feed" and frac < 1:
            augmented_train_examples, aug_acc, frac_used = self_feed(dir_to_save, labeled_examples, unlabeled_examples, val_ds, test_ds, text_field, label_field, classifier_params)
            aug_accs.append(aug_acc); aug_fracs.append(frac_used)
        elif aug_algo and frac < 1:
            augmented_train_examples, aug_acc, frac_used = augment(data_source, aug_algo, encoder_model, labeled_examples, unlabeled_examples, train_ds, text_field, label_field)
            aug_accs.append(aug_acc); aug_fracs.append(frac_used)
        else:
            augmented_train_examples = labeled_examples
            aug_accs.append(1)

        new_train_ds = data.Dataset(augmented_train_examples, {'x': text_field, 'y': label_field})
        new_datasets = (new_train_ds, val_ds, test_ds)
        mean, std, avg_p, avg_r, avg_f = repeat_ic(dir_to_save, text_field, label_field, new_datasets, classifier_params)
        means.append(mean); stds.append(std); ps.append(avg_p); rs.append(avg_r); fs.append(avg_f)

    print(f"FRAC '{frac}' RESULTS BELOW:")
    print(f'augmentation acc mean: {np.mean(aug_accs)}, augmentation acc std: {np.std(aug_accs)}, augmentation frac mean: {np.mean(aug_fracs)}')
    print(f'precision mean: {np.mean(ps)}, recall mean: {np.mean(rs)}, f1 mean: {np.mean(fs)}')
    print(f'class acc mean: {np.mean(means)}, class acc std: {np.std(means)}, mean of stds: {np.mean(stds)}')

    class_acc_mean, class_acc_std = np.mean(means), np.std(means)
    aug_acc_mean, aug_acc_std = np.mean(aug_accs), np.std(aug_accs)
    aug_frac_mean = np.mean(aug_fracs)
    p_mean, r_mean, f1_mean = np.mean(ps), np.mean(rs), np.mean(fs)
    p_std, r_std, f1_std = np.std(ps), np.std(rs), np.std(fs)

    return class_acc_mean, class_acc_std, aug_acc_mean, aug_acc_std, p_mean, p_std, r_mean, r_std, f1_mean, f1_std, aug_frac_mean


def augment(data_source, aug_algo, encoder_model, labeled_examples, unlabeled_examples, train_ds, text_field, label_field, normalise_encodings=True):
    if encoder_model.startswith('pretrained'):
        embedding_type = encoder_model.split('_')[1]
        res = encode_data_with_pretrained(data_source, train_ds, text_field, embedding_type, labeled_examples, unlabeled_examples)
    elif encoder_model.startswith('sts'):
        pass
    xs_l, ys_l, xs_u, ys_u, xs_u_unencoded = res

    if normalise_encodings:
        xs_l = np.array([x / np.linalg.norm(x) for x in xs_l])
        xs_u = np.array([x / np.linalg.norm(x) for x in xs_u])
    
    if aug_algo == "knn":
        classifications, num_correct = knn_classify(5, xs_l, ys_l, xs_u, ys_u, weights='uniform', distance_metric='euclidean')
    elif aug_algo == "my_label_prop":
        # perhaps I need to recur here and do the LP again (assuming correctness of new labels) if there's a sufficient number of unused unlabeled examples.
        lp = LabelProp(xs_l, ys_l, xs_u, 21) # hardcoded number of classes
        lp.propagate()
        classifications, indices = lp.classify(ys_u,threshold=True)
        ys_u = ys_u[indices]
        xs_u_unencoded = [xs_u_unencoded[idx] for idx in indices]
    elif aug_algo.startswith("label_prop"):
        type_ = aug_algo.split('__')[1]
        pd = {'kernel': 'knn', 'gamma': 20, 'n_neighbors': 1, 'alpha': None, 'max_iter': 100000, 'tol': 1e-3}
        classifications = label_prop_classify(xs_l, ys_l, xs_u, type_, pd)
    
    num_correct = np.sum(classifications == ys_u)
    frac_used = float(len(xs_u_unencoded)/len(xs_u))
    
    # unlabeled = [[text_field.vocab.itos[idx] for idx in sent] for sent in xs_u_unencoded] -- only need to do this if we idxs not strings
    new_labeled_data = [{'x': x, 'y': classifications[i]} for i,x in enumerate(xs_u_unencoded)]
    example_fields = {'x': ('x', text_field), 'y': ('y', label_field)}
    new_examples = [Example.fromdict(x, example_fields) for x in new_labeled_data]

    return labeled_examples + new_examples, float(num_correct / len(classifications)), frac_used


def encode_data_with_pretrained(data_source, train_ds, text_field, embedding_type, examples_l, examples_u):
    data_source_embeddings_path = f'./data/ic/{data_source}/{embedding_type}_embeddings.pkl'
    embeddings_file = Path(data_source_embeddings_path)
    
    if not embeddings_file.is_file():
        create_pretrained_embeddings(train_ds, text_field, data_source, embedding_type)
        embeddings_file = Path(data_source_embeddings_path)

    embeddings = pickle.load(embeddings_file.open('rb'))
    xs_l = np.array([embeddings[' '.join(eg.x)] for eg in examples_l])
    xs_u = np.array([embeddings[' '.join(eg.x)] for eg in examples_u])

    ys_l = np.array([eg.y for eg in examples_l])
    ys_u = np.array([eg.y for eg in examples_u])
    xs_u_unencoded = [eg.x for eg in examples_u]

    return xs_l, ys_l, xs_u, ys_u, xs_u_unencoded


def create_pretrained_embeddings(train_ds, text_field, data_source, embedding_type):
    if embedding_type == "glove":
        encoder = create_encoder(text_field.vocab, 300, "pool_max", *['max'])
        encoder.eval()
        sents = [torch.tensor([[text_field.vocab.stoi[t] for t in eg.x]]) for eg in train_ds.examples]
        embeddings = {}
        for eg, idxed in zip(train_ds.examples, sents):
            sent = ' '.join(eg.x)
            emb = encoder(idxed.reshape(-1, 1)).detach().squeeze(0).numpy()
            embeddings[sent] = emb
    else:
        encoder = {"bert": BertEmbeddings(), "elmo": ELMoEmbeddings()}[embedding_type]
        sents = [Sentence(' '.join(eg.x)) for eg in train_ds.examples]
        encoder.embed(sents)
        embs = np.array([torch.max(torch.stack([t.embedding for t in S]), 0)[0].detach().numpy() for S in sents])
        embeddings = {' '.join(eg.x): emb for eg, emb in zip(train_ds.examples, embs)}

    pickle.dump(embeddings, Path('./data/ic/{data_source}/{embedding_type}_embeddings.pkl').open('wb'))


def train_ic(dir_to_save, text_field, label_field, datasets, classifier_params):
    """
    To simplify training (connection to generic training utility with appropriate parameters).
    # ULTIMATELY, classifier_params should just be able to be passed to this train method along with other
    # stuff and THAT method is what sorts it all out... but this is fine for now.s
    """
    ps = classifier_params
    wrapper = traina(ps['model_name'], ps['encoder_model'], get_ic_data_iterators, IntentWrapper, dir_to_save,
                    nn.CrossEntropyLoss(), datasets, text_field, ps['bs'], ps['encoder_args'],
                    ps['layers'], ps['drops'], ps['lr'], frac=1, verbose=False)
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