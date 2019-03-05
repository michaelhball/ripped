import numpy as np
import pickle
import torch
import torch.nn as nn

from flair.data import Sentence
from flair.embeddings import BertEmbeddings, ELMoEmbeddings, FlairEmbeddings, StackedEmbeddings
from pathlib import Path
from torchtext import data
from torchtext.data.example import Example

from modules.data_iterators import get_ic_data_iterators
from modules.models import create_encoder
from modules.model_wrappers import IntentWrapper
from modules.utilities import repeat_trainer

from .knn import knn_classify


__all__ = ['repeat_augment_and_train']


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
    aug_accs = []
    means, stds = [], []
    ps, rs, fs = [], [], []
    for i in range(k):
        print(f"Augmentation run # {i+1}")
        examples = train_ds.examples
        np.random.shuffle(examples)
        cutoff = int(frac*len(examples))
        labeled_examples = examples[:cutoff]
        unlabeled_examples = examples[cutoff:]

        if aug_algo and frac < 1:
            augmented_train_examples, aug_acc = augment(data_source, aug_algo, encoder_model, labeled_examples, unlabeled_examples, train_ds, text_field, label_field)
            aug_accs.append(aug_acc)
        else:
            augmented_train_examples = labeled_examples
            aug_accs.append(1)

        new_train_ds = data.Dataset(augmented_train_examples, {'x': text_field, 'y': label_field})
        new_datasets = (new_train_ds, val_ds, test_ds)
        mean, std, avg_p, avg_r, avg_f = repeat_ic(dir_to_save, text_field, label_field, new_datasets, classifier_params)
        means.append(mean); stds.append(std); ps.append(avg_p); rs.append(avg_r); fs.append(avg_f)

    print(f"FRAC '{frac}' RESULTS BELOW:")
    print(f'augmentation acc mean: {np.mean(aug_accs)}, augmentation acc std: {np.std(aug_accs)}')
    print(f'precision mean: {np.mean(ps)}, recall mean: {np.mean(rs)}, f1 mean: {np.mean(fs)}')
    print(f'class acc mean: {np.mean(means)}, class acc std: {np.std(means)}, mean of stds: {np.mean(stds)}')

    class_acc_mean, class_acc_std = np.mean(means), np.std(means)
    aug_acc_mean, aug_acc_std = np.mean(aug_accs), np.std(aug_accs)
    p_mean, r_mean, f1_mean = np.mean(ps), np.mean(rs), np.mean(fs)
    p_std, r_std, f1_std = np.std(ps), np.std(rs), np.std(fs)

    return class_acc_mean, class_acc_std, aug_acc_mean, aug_acc_std, p_mean, p_std, r_mean, r_std, f1_mean, f1_std


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
    elif aug_algo == "lp":
        pass
    
    # unlabeled = [[text_field.vocab.itos[idx] for idx in sent] for sent in xs_u_unencoded] -- only need to do this if we idxs not strings
    new_labeled_data = [{'x': x, 'y': classifications[i]} for i,x in enumerate(xs_u_unencoded)]
    example_fields = {'x': ('x', text_field), 'y': ('y', label_field)}
    new_examples = [Example.fromdict(x, example_fields) for x in new_labeled_data]

    return labeled_examples + new_examples, float(num_correct / len(classifications))


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