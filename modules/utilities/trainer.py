"""
Utility functions for managing training processes.
"""
import numpy as np
import time
import torch

from sklearn.model_selection import ParameterGrid
from torchtext import data, vocab
from tqdm import tqdm


__all__ = ['grid_search', 'repeat_trainer']


def grid_search(get_iter_func, wrapper_class, saved_models, datasets, param_grid, encoder_type, encoder_args, layers, text_field, label_field, frac=1, k=10, verbose=True):
    """
    Perform grid search over given parameter options.
    Args:
        saved_models (str): directory to save temporary models
        datasets (tuple(Dataset)): torchtext datasets w train/val/test data
        param_grid (dict): param values: {str: list(values)}
        encoder_type (str): encoder type to use
        layers (list(int)): list of feedforward layers for classifier
        text_field (Field): torchtext field representing x values
        label_field (LabelField): torchtext labelfield representing y values
        frac (float): fraction of training data to use
        k (int): number of training runs to perform for each trial
    Returns:
        List of trials sorted by mean accuracy (each trial performs k training runs)
    """
    results = []
    for i, params in enumerate(ParameterGrid(param_grid)):
        lr = params['lr']
        bs = params['bs']
        drops = [params['drop1'], params['drop2']]
        loss_func = nn.CrossEntropyLoss() # hardcoded
        mean, std = repeat_trainer(get_iter_func, wrapper_class, saved_models, loss_func, datasets, text_field, label_field, bs, encoder_args, layers, drops, lr, frac=frac, k=k, verbose=verbose)
        print("-------------------------------------------------------------------------------------------------------------------------------------------")
        print(i, params, mean, std)
        print("-------------------------------------------------------------------------------------------------------------------------------------------")
        results.append({'mean': mean, 'std': std, 'params': params})

    return(reversed(sorted(results, key=lambda x: x['mean'])))


def repeat_trainer(model_name, encoder_model, get_iter_func, wrapper_class, saved_models, loss_func, datasets, text_field, label_field, bs, encoder_args, layers, drops, lr, frac=1, k=10, verbose=True):
    """
    Function to perform multiple training runs (for model validation).
    Args:
        ...
        saved_models (str): directory to save temporary models
        loss_func (): pytorch loss function for training
        datasets (tuple(Dataset)): torchtext datasets for train/val/test data
        text_field (Field): torchtext field representing x values
        label_field (LabelField): torchtext labelfield representing y values
        bs (int): batch_size
        layers (list(int)): list of feedforward layers for classifier
        drops (list(float)): list of dropouts to apply to layers
        lr (float): learning rate for training
        frac (float): fraction of training data to use
        k (int): number of training runs to perform
        verbose (bool): for printing
    Returns:
        mean and standard deviation of training run accuracy
    """
    if verbose:
        print(f"-------------------------  Performing {k} Training Runs -------------------------")
        start_time = time.time()

    train_ds, val_ds, test_ds = datasets
    it = tqdm(range(k), total=k) if verbose else range(k)
    for i in it:
        name = f'{model_name}_{i}'
        examples = train_ds.examples
        np.random.shuffle(examples)
        new_train_ds = data.Dataset(examples[:int(len(examples)*frac)], {'x': text_field, 'y': label_field})
        train_di, val_di, test_di = get_iter_func(new_train_ds, val_ds, test_ds, (bs,bs,bs))
        wrapper = wrapper_class(name, saved_models, 300, text_field.vocab, encoder_model, train_di, val_di, test_di, encoder_args, layers=layers, drops=drops)
        opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=lr, betas=(0.7, 0.999), weight_decay=0)
        train_losses, test_losses = wrapper.train(loss_func, opt_func, verbose=False)

    if verbose:
        elapsed_time = time.time() - start_time
        print("{0} training runs completed in {1}".format(k, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    
    accuracies, precisions, recalls, f1s = [], [], [], []
    for i in range(k):
        name = f'{model_name}_{i}'
        train_di, val_di, test_di = get_iter_func(train_ds, val_ds, test_ds, (bs,bs,bs))
        wrapper = wrapper_class(name, saved_models, 300, text_field.vocab, encoder_model, train_di, val_di, test_di, encoder_args, layers=layers, drops=drops)
        accuracies.append(wrapper.test_accuracy(load=True))
        p, r, f, s = wrapper.test_precision_recall_f1(load=True)
        precisions.append(np.mean(p))
        recalls.append(np.mean(r))
        f1s.append(np.mean(f))
    
    return np.mean(accuracies), np.std(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1s)
