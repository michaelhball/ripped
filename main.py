import argparse
import csv
import gensim.downloader as api
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import spacy
import time
import torch
import torch.nn as nn
import torch.nn.functional as Fs

from flair.data import Sentence
from flair.embeddings import BertEmbeddings, ELMoEmbeddings, FlairEmbeddings, StackedEmbeddings
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from torchtext import data, vocab
from torchtext.data.example import Example
from tqdm import tqdm

from modules.data_iterators import *
from modules.data_readers import IntentClassificationDataReader, SentEvalDataReader, STSDataReader
from modules.models import *
from modules.model_wrappers import BaselineWrapper, DownstreamWrapper, IntentWrapper, MTLWrapper, ProbingWrapper, STSWrapper
from modules.ssl import knn_classify
from modules.utilities import *

from results import all_results


parser = argparse.ArgumentParser(description='PyTorch Dependency-Parse Encoding Model')
parser.add_argument('--word_embedding_source', type=str, default='glove-wiki-gigaword-50', help='word embedding source to use')
parser.add_argument('--word_embedding', type=str, default='glove_50', help='name of default word embeddings to use')
parser.add_argument('--saved_models', type=str, default='./models', help='directory to save/load models')
parser.add_argument('--encoder_model', type=str, default='pos_tree', help='pos_lin/pos_tree/dep_tree')
parser.add_argument('--predictor_model', type=str, default='mlp', help='mlp / cosine_sim')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--model_name', type=str, default='999', help='name of model to train')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
parser.add_argument('--subtask', type=str, default='none', help='sub-task within whichever specified task')
parser.add_argument('--baseline_type', type=str, default='pool', help='type of baseline model')
parser.add_argument('--lr', type=float, default=6e-4, help='learning rate for whichever model is being trained')
parser.add_argument('--wd', type=float, default=0, help='L2 regularization for training')
parser.add_argument('--frac', type=float, default=1, help='fraction of training data to use')
args = parser.parse_args()


def create_data_source_embeddings(train_ds, text_field, data_source, embedding_type):
    if embedding_type == "glove":
        encoder = create_encoder(text_field.vocab, 300, "pool_max", *['max'])
        encoder.eval()
        sents = [torch.tensor([[text_field.vocab.stoi[t] for t in eg.x]]) for eg in train_ds.examples]
        embeddings = {}
        for eg, idxed in zip(train_ds.examples, sents):
            sent = ' '.join(eg.x)
            emb = encoder(idxed.reshape(-1, 1)).detach().squeeze(0).numpy()
            embeddings[sent] = embs
    elif embedding_type == "bert":
        encoder = BertEmbeddings()
        sents = [Sentence(' '.join(eg.x)) for eg in train_ds.examples]
        encoder.embed(sents)
        b_embeddings = np.array([torch.max(torch.stack([t.embedding for t in S]), 0)[0].detach().numpy() for S in sents])
        embeddings = {' '.join(eg.x): emb for eg, emb in zip(train_ds.examples, b_embeddings)}

    pickle.dump(embeddings, Path('./data/ic/{data_source}/{embedding_type}_embeddings.pkl').open('wb'))


def encode_data_with_pretrained(train_ds, text_field, embedding_type, examples_l, examples_u):
    data_source_embeddings_path = f'./data/ic/{data_source}/{embedding_type}_embeddings.pkl'
    embeddings_file = Path(data_source_embeddings_path)
    
    if not embeddings_file.is_file():
        create_data_source_embeddings(train_ds, text_field, data_source, embedding_type)
        embeddings_file = Path(data_source_embeddings_path)

    embeddings = pickle.load(embeddings_file.open('rb'))
    xs_l = np.array([embeddings[' '.join(eg.x)] for eg in examples_l])
    xs_u = np.array([embeddings[' '.join(eg.x)] for eg in examples_u])

    ys_l = np.array([eg.y for eg in examples_l])
    ys_u = np.array([eg.y for eg in examples_u])
    xs_u_unencoded = [eg.x for eg in examples_u]

    return xs_l, ys_l, xs_u, ys_u, xs_u_unencoded


def augment(aug_algo, encoder_model, labeled_examples, unlabeled_examples, train_ds, text_field, label_field, normalise_encodings=True):
    if encoder_model.startswith('pretrained'):
        embedding_type = encoder_model.split('_')[1]
        res = encode_data_with_pretrained(train_ds, text_field, embedding_type, labeled_examples, unlabeled_examples)
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


def repeat_augment_and_train(dir_to_save, aug_algo, encoder_model, datasets, text_field, label_field, frac, classifier_params, k=5):
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
            augmented_train_examples, aug_acc = augment(aug_algo, encoder_model, labeled_examples, unlabeled_examples, train_ds, text_field, label_field)
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


def get_results(algorithm, data_source, classifier, encoder=None, similarity_measure=None):
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
    if algorithm is 'supervised':
        results_name = f'{data_source}__supervised__{classifier}'
    else:
        results_name = f'{data_source}_{algorithm}__{encoder}__{similarity_measure}__{classifier}'
    
    return all_results[results_name]


####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################################################################################################################


if __name__ == "__main__":
    # ss_methods = {
    #     "knn-bert": {
    #         "algorithm": "knn_all",
    #         "encoder": "bert",
    #         "similarity": "cosine",
    #         "colour": "b-"
    #     }
    # }
    # data_source = 'chat'
    # classifier = "pool_max"
    # to_plot = "class_acc"
    # plot_against_supervised(ss_methods, data_source, classifier, get_results, to_plot=to_plot, display=True, save_file=None)
    # assert(False)

    embedding_dim = int(args.word_embedding.split('_')[1])
    layers = [2*embedding_dim, 250, 1]
    drops = [0, 0]
    sick_train_data = './data/sts/sick/train_data'
    sick_test_data = './data/sts/sick/test_data'
    train_data_raw = pickle.load(Path('./data/sts/sick/train.pkl').open('rb'))
    test_data_raw = pickle.load(Path('./data/sts/sick/test.pkl').open('rb'))
    # di_suffix = {"pos_lin": "og", "pos_tree": "trees", "dep_tree": "trees"}
    # train_di = EasyIterator(f'{sick_train_data}_{args.word_embedding}_{di_suffix[args.encoder_model]}.pkl')
    # test_di = EasyIterator(f'{sick_test_data}_{args.word_embedding}_{di_suffix[args.encoder_model]}.pkl', randomise=False)

    if args.task.startswith("propagater"):
        # params for everything
        t = args.task.split("_")
        task = t[0]; data_source = t[1]
        intents = pickle.load(Path(f'./data/ic/{data_source}/intents.pkl').open('rb'))
        C = len(intents)
        EMB_DIM = 300

        # create datasets/vocab
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        FRAC = args.frac
        train_ds, val_ds, test_ds = IntentClassificationDataReader(f'./data/ic/{data_source}/', '_tknsd.pkl', TEXT, LABEL).read()
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings)

        # repeatedly augment dataset (using different labeled fraction) and repeatedly train classifier for each.
        
        # standard intent classifier for testing different learning methods.
        classifier_params = {
            'model_name': 'test',
            'encoder_model': 'pool_max',
            'encoder_args': ['max'],
            'emb_dim': EMB_DIM,
            'layers': [EMB_DIM, 100, C],
            'drops': [0, 0],
            'bs': 64,
            'lr': 6e-4
        }

        # run augmentation trials
        aug_algo = 'knn' # 'knn'|'label_propagation'|None
        dir_to_save = f'{args.saved_models}/ic/{data_source}'
        class_acc_means, class_acc_stds, aug_acc_means, aug_acc_stds = [],[],[],[]
        p_means, p_stds, r_means, r_stds, f1_means, f1_stds = [],[],[],[],[],[]
        for FRAC in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
            class_acc_mean, class_acc_std, aug_acc_mean, aug_acc_std, p_mean, p_std, r_mean, r_std, f1_mean, f1_std = repeat_augment_and_train(dir_to_save, aug_algo, args.encoder_model, (train_ds, val_ds, test_ds), TEXT, LABEL, FRAC, classifier_params, k=5)
            class_acc_means.append(class_acc_mean); class_acc_stds.append(class_acc_std)
            aug_acc_means.append(aug_acc_mean); aug_acc_stds.append(aug_acc_std)
            p_means.append(p_mean); r_means.append(r_mean); f1_means.append(f1_mean)
            p_stds.append(p_std); r_stds.append(r_std); f1_stds.append(f1_std)
        print(class_acc_means)
        print(class_acc_stds)
        print(aug_acc_means)
        print(aug_acc_stds)
        print(p_means)
        print(p_stds)
        print(r_means)
        print(r_stds)
        print(f1_means)
        print(f1_stds)


    elif args.task.startswith("propagate"):
        data_source = args.task.split('_')[1]
        intents = pickle.load(Path(f'./data/ic/{data_source}/intents.pkl').open('rb'))
        C = len(intents)
        BS = 64
        EMB_DIM = 300
        FRAC = args.frac

        # get labeled & unlabeled data iterators
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        train_ds, val_ds, test_ds = IntentClassificationDataReader(f'./data/ic/{data_source}/', '_tknsd.pkl', TEXT, LABEL).read()
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings)
        # CONVERT VOCAB FROM OLD TO NEW HERE SOMEHOW... IF WE WANT TO BE ABLE TO USE OUR PRETRAINED MODELS THAT IS.
        # can I just override update itos sheit somehow and it should work automatically??
        # refer to ULMFiT stuff
        
        # COSINE SIMILARITY: (otherwise we need an entire predictor distance measure thing...)

        # create encoder
        if args.encoder_model.startswith('pool'):
            encoder_args = [args.encoder_model.split('_')[1]] # pool_type
        elif args.encoder_model == 'lstm':
            encoder_args = [EMB_DIM, 1, False, False] # hidden_size, num_layers, bidirectional, fine_tune_we
        elif args.encoder_model.startswith("infersent"):
            # batch_size, encoding_dim, pool_type, dropout, version, state_dict_path, w2v_path, sentences, fine_tune
            pool_type = args.encoder_model.split('_')[1]
            sd_path = './pretrained_models/infersent/infersent1.pkl'
            w2v_path = './data/glove.840B.300d.txt'
            sentences = [' '.join(eg.x) for eg in train_ds.examples]
            encoder_args = [BS, 2048, pool_type, 0, 1, sd_path, w2v_path, sentences, False]
        
        encoder = create_encoder(TEXT.vocab, EMB_DIM, args.encoder_model, *encoder_args)
        
        # ''' # Only need the following if we are loading a model that I have trained... i.e. purely not InferSent at the moment.
        # enc_state_dict_path = f'{args.saved_models}/sts/stsbenchmark/lstm_0.709_encoder.pt'
        # encoder.load_state_dict(torch.load(enc_state_dict_path))
        # encoder.eval(); encoder.training = False
        # # WE HAVE TO CONVERT WEIGHTS FROM THE EMBEDDING LAYER TO THAT NEEDED IN INTENT CLASSIFICATION TASK WON'T I... I.E.
        # # WE HAVE A NEW VOCAB HERE...
        # print(encoder.state_dict().keys())
        # assert(False)
        # '''

        propagation_method = args.task.split('_')[2] # knn, lp
        augmented_train_examples = augment_data_all(propagation_method, encoder, args.encoder_model, train_ds, FRAC, TEXT, LABEL, BS, True)
        print(len(augmented_train_examples))

            #################################
            # KMEANS CLUSTERING
            #################################

            # from sklearn.cluster import KMeans
            # clusterer = KMeans(n_clusters=C, random_state=0).fit(Xs)
            # labels = clusterer.labels_
            
            # # find the indices of all data points in each cluster
            # clusters = {i: [] for i in range(C)}
            # for i, l in enumerate(labels):
            #     clusters[l].append(i)

            # # find the most common true label of all data points in each cluster
            # cluster_labels = {}
            # for l, idxs in clusters.items():
            #     lst = [Ys[i] for i in idxs]
            #     cluster_labels[l] = max(set(lst), key=lst.count)

            # # get accuracy
            # converted_labels = [cluster_labels[l] for l in labels]
            # accuracies.append(np.sum(converted_labels == Ys) / len(Ys))

    elif args.task.startswith("ic"):
        t = args.task.split('_'); task = t[0]; data_source = t[1]
        intents = pickle.load(Path(f'./data/ic/{data_source}/intents.pkl').open('rb'))
        C = len(intents)
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        BS = 64 if args.encoder_model.startswith("pool") else 128
        EMB_DIM, HID_DIM = 300, 100
        FRAC = args.frac
        LR = args.lr
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
            opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=LR, betas=(0.7,0.999), weight_decay=args.wd)
            train_losses, val_losses = wrapper.train(loss_func, opt_func)

    elif args.task.startswith("sts"):
        C = 1
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.float, use_vocab=False)
        BS = 64
        LR = args.lr
        EMB_DIM = 300
        HID_DIM = 300
        layers = [2*EMB_DIM, HID_DIM, C]
        drops = [0, 0]
        
        data_source = args.task.split('_')[1]
        saved_models_path = f'{args.saved_models}/sts/{data_source}'
        train_file = f'./data/sts/{data_source}/train_tknsd.pkl' # './data/sts/both_train_tknsd.pkl'
        # train_file = './data/sts/both_train_tknsd.pkl'
        val_file = f'./data/sts/{data_source}/val_tknsd.pkl'
        test_file = f'./data/sts/{data_source}/test_tknsd.pkl' # './data/sts/both_test_tknsd.pkl'
        train_ds, val_ds, test_ds = STSDataReader(train_file, test_file, test_file, TEXT, LABEL).read() # using test data for validation
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings)

        if args.encoder_model.startswith('pool'):
            encoder_args = [encoder_model.split('_')[1]] # pool_type
        elif args.encoder_model == 'lstm':
            encoder_args = [EMB_DIM, 1, False, False] # hidden_size, num_layers, bidirectional, fine_tune_we
        elif args.encoder_model == "infersent":
            encoder_args = []
        else:
            print(f'there are no classes set up for encoder model "{args.encoder_model}"')
        
        if args.subtask == "train":
            train_di, val_di, test_di = get_sts_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
            loss_func = nn.MSELoss()
            wrapper = STSWrapper(args.model_name,saved_models_path,EMB_DIM,TEXT.vocab,args.encoder_model,args.predictor_model,train_di,val_di,test_di,layers,drops,*encoder_args)
            opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=LR, betas=(0.7,0.999), weight_decay=args.wd)
            train_losses, val_losses, correlations = wrapper.train(loss_func, opt_func)
        elif args.subtask == "test":
            train_di, val_di, test_di = get_sts_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
            wrapper = STSWrapper(args.model_name,saved_models_path,EMB_DIM,TEXT.vocab,args.encoder_model,args.predictor_model,train_di,val_di,test_di,layers,drops,*encoder_args)
            p,s = wrapper.test_correlation(load=True)
            print(f'pearson: {round(p[0],3)}, spearman: {round(s[0],3)}')


    elif args.task =="infersent":

        intents = pickle.load(Path('./data/chat_ic/intents.pkl').open('rb'))
        C = len(intents)
        TEXT = data.Field(sequential=True, use_vocab=False) # REMEMBER TO CHANGE THIS BACK IF I NEED TO.
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        BS = 128
        FRAC = args.frac
        LR = args.lr
        layers = [300, 100, C]
        drops = [0, 0]

        import nltk
        from pretrained_models.infersent.models import InferSent
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        model = InferSent(params_model)
        model.load_state_dict(torch.load('./pretrained_models/infersent/infersent1.pkl'))
        model.set_w2v_path('./data/glove.840B.300d.txt')
        
        # build vocab
        train_ds, val_ds, test_ds = IntentClassificationDataReader('./data/chat_ic/', '_tknsd.pkl', TEXT, LABEL).read()
        sentences = [' '.join(eg.x) for eg in train_ds.examples]
        model.build_vocab(sentences, tokenize=False)

        # print(model.word_vec.keys())
        # I'll have to build a vocab here to satisfy Torchtext, and then use itos to convert from the
        # integers to the words before passing into InferSent?? this sounds like a massive pain in the ass...
        # but seems like it might be the best way... i.e. the model itself has a 'vocab' which is really just
        # its word2vec, and then TEXT also has a VOCAB that's used to create the iterator of the right size etc. etc.
        # I could try using a normal iterator (i.e. not bucket, setting the sort key param)???
        # try this before implementing the massive redundant thing

        # train
        train_di, val_di, test_di = get_ic_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
        print(next(iter(train_di)))
        print(next(iter(val_di)))
        print(next(iter(test_di)))
        # need to see if this is still text tokens rather than token indices... we don't want TEXT to create a vocab here.
        assert(False)
        loss_func = nn.CrossEntropyLoss()
        wrapper = IntentWrapper(args.model_name, f'{args.saved_models}/chat_ic', 300, model, args.encoder_model, train_di, val_di, test_di, layers=layers, drops=drops)
        opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=LR, betas=(0.7,0.999), weight_decay=args.wd)
        train_losses, val_losses = wrapper.train(loss_func, opt_func)



    # elif args.task == "chat_ic":
    #     # fracs = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]
    #     # we_accs = [0.848,0.828,0.822,0.81,0.795,0.786,0.764,0.722,0.712,0.619,0.436]
    #     # we_cis = [0.005901542,0.009797332,0.007560146,0.010067337,0.01172594,0.011263074,
    #     #         0.013423116,0.012767389,0.014310276,0.02005753,0.064415528]
    #     # lstm_accs = [0.916,0.9,0.892,0.872,0.846,0.845,0.819,0.78,0.727,0.577,0.425]
    #     # lstm_cis = [0.006171548,0.006827275,0.008177301,0.008215873,0.008177301,0.008755883,
    #     #         0.009643043,0.016277457,0.019556092,0.017550339,0.029932006]
    #     # we_f1s = [0.789,0.752,0.750,0.724,0.705,0.690,0.669,0.603,0.583,0.469,0.281]
    #     # lstm_f1s = [0.867,0.856,0.852,0.831,0.755,0.730,0.725,0.625,0.549,0.364,0.239]
    #     # # plt.plot(fracs, we_f1s, 'r-')
    #     # # plt.plot(fracs, lstm_f1s, 'b-')
    #     # plt.errorbar(fracs,we_accs,yerr=we_cis,fmt='r-',ecolor='black',elinewidth=0.5,capsize=1,label='we_pool')
    #     # plt.errorbar(fracs,lstm_accs,yerr=lstm_cis,fmt='b-', ecolor='blue',elinewidth=0.5,capsize=1,label='lstm')
    #     # plt.xticks([0.1*i for i in range(0,11)])
    #     # plt.title('F1 scores w different fractions of training data')
    #     # plt.xlabel('fraction of training data'); plt.ylabel('F1')
    #     # plt.legend()
    #     # plt.show()
    #     # assert(False)

    #     intents = pickle.load(Path('./data/chat_ic/intents.pkl').open('rb'))
    #     C = len(intents)
    #     TEXT = data.Field(sequential=True)
    #     LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
    #     BS = 64 if args.encoder_model == "we_pool" else 128
    #     FRAC = args.frac
    #     LR = args.lr
    #     layers = [300, 100, C]
    #     drops = [0, 0] if args.encoder_model == "we_pool" else [0.1, 0.2]

    #     train_ds, val_ds, test_ds = IntentClassificationDataReader('./data/chat_ic/', '_tknsd.pkl', TEXT, LABEL).read()
    #     glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
    #     TEXT.build_vocab(train_ds, vectors=glove_embeddings) # 461

    #     if args.subtask == "grid_search":
    #         param_grid = {'lr': [3e-3, 6e-3, 1e-2], 'drop1': [0, 0.1, 0.2], 'drop2': [0, 0.1, 0.2], 'bs': [128]}
    #         results = grid_search(f'{args.saved_models}/chat_ic', train_ds, val_ds, test_ds,
    #                 param_grid, args.encoder_model, layers, TEXT, LABEL, k=5, verbose=False)
    #         print(results)
    #     elif args.subtask == "repeat_train":
    #         for frac in (0.9, 0.8):
    #             train_ds, val_ds, test_ds = IntentClassificationDataReader('./data/chat_ic/', '_tknsd.pkl', TEXT, LABEL).read()
    #             glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
    #             TEXT.build_vocab(train_ds, vectors=glove_embeddings) # 461
    #             FRAC = frac
    #             loss_func = nn.CrossEntropyLoss()
    #             mean, std = repeat_trainer(f'{args.saved_models}/{args.task}', loss_func, train_ds, val_ds, test_ds,
    #                     TEXT, LABEL, BS, layers, drops, LR, FRAC, k=10, verbose=True)
    #             print(mean, std)
    #             print(f"frac^{frac}")
    #     elif args.subtask == "train":
    #         train_di, val_di, test_di = get_ic_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
    #         loss_func = nn.CrossEntropyLoss()
    #         wrapper = IntentWrapper(args.model_name, f'{args.saved_models}/chat_ic', 300, TEXT.vocab, args.encoder_model, train_di, val_di, test_di, layers=layers, drops=drops)
    #         opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=LR, betas=(0.7,0.999), weight_decay=args.wd)
    #         train_losses, val_losses = wrapper.train(loss_func, opt_func)


    # elif args.task == "snipsnlu":
    #     # accuracies = [0.951,0.941,0.865,0.81,0.606,0.245]
    #     # fracs = [1,0.1,0.01,0.005,0.002,0.001]
    #     # cis = [0.001146009,0.008595071,0.011460094,0.02062817,0.096264792,0.111735919]
    #     # # plt.plot(fracs, accuracies, color='black')
    #     # plt.xlabel('fraction of training data'); plt.ylabel('accuracy')
    #     # plt.xscale('log')
    #     # plt.yticks([0.1*i for i in range(11)])
    #     # plt.title('Accuracy using different fractions of training data')
    #     # plt.errorbar(fracs,accuracies,yerr=cis,fmt='r-',ecolor='black',elinewidth=0.5,capsize=1)
    #     # plt.show()
    #     # assert(False)

    #     intents = ['add_to_playlist', 'book_restaurant', 'get_weather', 'play_music', 'rate_book', 'search_creative_work', 'search_screening_event']
    #     C = len(intents)
    #     TEXT = data.Field(sequential=True)
    #     LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
    #     BS = 64 if args.encoder_model == "we_pool" else 128
    #     FRAC = args.frac
    #     layers = [300, 100, C]
    #     drops = [0, 0]

    #     # get datasets & create vocab
    #     train_ds, val_ds, test_ds = IntentClassificationDataReader('./data/snipsnlu/', '_tknsd.pkl', TEXT, LABEL).read()
    #     glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
    #     TEXT.build_vocab(train_ds, vectors=glove_embeddings) # 10196

    #     if args.subtask == "grid_search":
    #         param_grid = {'lr': [1e-3, 6e-4, 3e-3, 6e-3], 'drop1': [0, 0.1], 'drop2': [0, 0.1], 'bs': [128, 64]}
    #         ic_grid_search(train_ds, val_ds, test_ds, param_grid, args.encoder_model, layers, TEXT, LABEL)
        
    #     elif args.subtask == "repeat_train":
    #         loss_func = nn.CrossEntropyLoss()
    #         print(repeat_trainer(loss_func, train_ds, val_ds, test_ds, TEXT, LABEL, BS, layers, drops, args.lr, FRAC, k=10))

    #     elif args.subtask == "train":
    #         train_di, val_di, test_di = get_ic_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
    #         wrapper = IntentWrapper(args.model_name, f'{args.saved_models}/intent_class', 300, TEXT.vocab, args.encoder_model, train_di, val_di, test_di, layers=layers, drops=drops)
    #         loss_func = nn.CrossEntropyLoss()
    #         opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=args.lr, betas=(0.7, 0.999), weight_decay=args.wd, amsgrad=False)
    #         train_losses, test_losses = wrapper.train(loss_func, opt_func)
    #         # print(wrapper.repeat_trainer(loss_func, torch.optim.Adam, args.lr, (0.7,0.999), args.wd, k=20))


    elif args.task == "train_sts_benchmark":
        # import pandas as pd
        # df = pd.read_csv('./data/stsbenchmark/sts-train.csv', index_col=None, sep='\t', header=None, names=['to_delete', 'to_delete', 'to_delete', 'id', 'similarity', 's1', 's2'])
        # df = df.drop(columns=['to_delete', 'to_delete.1', 'to_delete.2', 'id'])
        # df.drop_duplicates()
        # df.reset_index(drop=True)
        # df = df[pd.notnull(df['s2'])]

        # import csv
        # test_data = []
        # with open('./data/stsbenchmark/sts-test.csv', 'r') as f:
        #     csv_reader = csv.reader(f)
        #     for r in csv_reader:
        #         row = list(filter(None, r))
        #         if len(row) == 1:
        #             test_data.append(row[0].split('\t'))
        #         else:
        #             test_data.append(''.join(row).split('\t'))
        # test_data = [r[4:7] for r in test_data]

        # sentences = []
        # for row in df.iterrows():
        #     sentences += [row[1]['s1'], row[1]['s2']]
        # for row in test_data:
        #     sentences += [row[1], row[2]]
        # pickle.dump(sentences, Path('./data/stsbenchmark/sentences.pkl').open('wb'))

        # import nltk
        # from pretrained_models.infersent.models import InferSent
        # params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
        #                 'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        # model = InferSent(params_model)
        # model.load_state_dict(torch.load('infersent1.pkl'))
        # model.set_w2v_path('./data/glove.840B.300d.txt')
        # model.build_vocab(sentences, tokenize=True)
        # embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
        # pickle.dump(embeddings, Path('./data/stsbenchmark/embeddings.pkl').open('wb'))

        # sentences = pickle.load(Path('./data/stsbenchmark/sentences.pkl').open('rb'))
        # embeddings = pickle.load(Path('./data/stsbenchmark/embeddings.pkl').open('rb'))
        # s_enc_map = {}
        # for i, s in enumerate(sentences):
        #     if s not in s_enc_map:
        #         s_enc_map[s] = embeddings[i]
        # pickle.dump(s_enc_map, Path('./data/stsbenchmark/encoding_map.pkl').open('wb'))
        
        # enc = pickle.load(Path('./data/stsbenchmark/encoding_map.pkl').open('rb'))
        # train_data = []
        # for x in df.iterrows():
        #     train_data.append([enc[x[1]['s1']], enc[x[1]['s2']], [float(x[1]['similarity'])]])
        # pickle.dump(np.array(train_data), Path('./data/stsbenchmark/train_infersent.pkl').open('wb'))
        # test_data = np.array([[enc[x[1]], enc[x[2]], [float(x[0])]] for x in test_data])
        # pickle.dump(test_data, Path('./data/stsbenchmark/test_infersent.pkl').open('wb'))

        # train_data_1 = pickle.load(Path('./data/stsbenchmark/train_infersent.pkl').open('rb'))
        # train_data_2 = pickle.load(Path('./data/sick/train_infersent_2.pkl').open('rb'))
        # train_data = np.concatenate([train_data_1, train_data_2])
        # pickle.dump(train_data, Path('./data/sts_train_all.pkl').open('wb'))
        # test_data_1 = pickle.load(Path('./data/stsbenchmark/test_infersent.pkl').open('rb'))
        # test_data_2 = pickle.load(Path('./data/sick/test_infersent_2.pkl').open('rb'))
        # test_data = np.concatenate([test_data_1, test_data_2])
        # pickle.dump(test_data, Path('./data/sts_test_all.pkl').open('wb'))

        # train_di = STSDataIterator('./data/stsbenchmark/train_infersent.pkl', batch_size=50, randomise=True)
        # test_di = STSDataIterator('./data/stsbenchmark/test_infersent.pkl', randomise=False)
        # train_di = STSDataIterator('./data/sts_train_all.pkl', batch_size=50, randomise=True)
        # test_di = STSDataIterator('./data/sts_test_all.pkl', randomise=False)
        train_di = STSDataIterator('./data/sick/train_infersent_2.pkl', batch_size=50, randomise=True)
        test_di = STSDataIterator('./data/sick/test_infersent_2.pkl', randomise=False)
        layers, drops = [2*4096, 1024, 1], [0.3, 0]
        predictor = STSWrapper(args.model_name, args.saved_models, train_di, test_di, "pretrained", layers=layers, drops=drops)
        loss_func = nn.MSELoss()
        # opt_func = torch.optim.Adam(predictor.model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
        # opt_func = torch.optim.Adam(predictor.model.parameters(), lr=0.01, weight_decay=args.wd, amsgrad=False)
        opt_func = torch.optim.SGD(predictor.model.parameters(), lr=0.01)
        train_losses, test_losses = predictor.train(loss_func, opt_func, args.num_epochs)




    elif args.task == "train_sts_sick":

        # sentences = []
        # for s1,s2,_ in train_data_raw:
        #     sentences.append(s1)
        #     sentences.append(s2)
        # for s1,s2,_ in test_data_raw:
        #     sentences.append(s1)
        #     sentences.append(s2)
        # pickle.dump(sentences, Path('sentences.pkl').open('wb'))

        # import nltk
        # from pretrained_models.infersent.models import InferSent
        # params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
        #                 'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        # model = InferSent(params_model)
        # model.load_state_dict(torch.load('infersent1.pkl'))
        # model.set_w2v_path('./data/glove.840B.300d.txt')
        # model.build_vocab(sentences, tokenize=True)
        # embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
        
        # pickle.dump(sentences, Path('9k_sentences.pkl').open('wb'))
        # pickle.dump(embeddings, Path('embeddings.pkl').open('wb'))
        

        # sentences = pickle.load(Path('sentences.pkl').open('rb'))
        # embeddings = pickle.load(Path('embeddings.pkl').open('rb'))
        # s_enc_map = {}
        # for i, s in enumerate(sentences):
        #     if s not in s_enc_map:
        #         s_enc_map[s] = embeddings[i]
        # pickle.dump(s_enc_map, Path('encoding_map.pkl').open('wb'))
        
        # encoding_map = pickle.load(Path('encoding_map.pkl').open('rb'))


        train_di = STSDataIterator('./data/sts/sick/train_infersent_2.pkl', batch_size=64, randomise=True)
        test_di = STSDataIterator('./data/sts/sick/test_infersent_2.pkl', randomise=False)
        layers, drops = [2*4096, 512, 1], [0, 0, 0]
        wrapper = STSWrapper(args.model_name, args.saved_models, train_di, test_di, "pretrained", predictor_model="nn", layers=layers, drops=drops)
        loss_func = nn.MSELoss()
        opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
        train_losses, test_losses = wrapper.train(loss_func, opt_func, args.num_epochs)


    elif args.task == "elmo":
        
        sentences = ['hey my name is Nick and I have a penis']
        
        # from mosestokenizer import MosesTokenizer, MosesDetokenizer
        # tokeniser = MosesTokenizer()
        # tknzd = tokeniser(sentences[0])
        # tokeniser.close()
        # print(tknzd)
        
        import spacy
        nlp = spacy.load('en')
        tknzd = [[t.text for t in nlp(s)] for s in sentences]

        from allennlp.commands.elmo import ElmoEmbedder
        from allennlp.modules.elmo import Elmo, batch_to_ids
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        
        # elmo = ElmoEmbedder(options_file, weight_file)
        # embeddings = elmo.embed_sentence(tknzd[0])
        # print(embeddings.shape)

        elmo = Elmo(options_file, weight_file, 1, dropout=0)
        character_ids = batch_to_ids(tknzd)
        embeddings = elmo(character_ids)
        print(embeddings['elmo_representations'][0].shape)
        # assuming I do one sentence at a time, this gives [seq_len, 1024] and then I can pool over these?


    elif args.task == "train_baseline":
        loss_func = nn.MSELoss()
        train_data = f'{sick_train_data}_{args.word_embedding}_baseline.pkl'
        test_data = f'{sick_test_data}_{args.word_embedding}_baseline.pkl'
        predictor = BaselineWrapper(args.model_name, args.saved_models, train_data, test_data, layers, drops, args.baseline_type, embedding_dim=embedding_dim)
        # opt_func = torch.optim.Adam(predictor.model.parameters(), lr=args.lr)
        predictor.train(loss_func, opt_func, args.num_epochs)
        predictor.save()
        p, s, i = predictor.test_correlation()
        print(p[0], s[0])

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

    elif args.task == "test_sst":
        train_data = f'./data/sst/train_data_{args.word_embedding}_og.pkl'
        test_data = f'./data/sst/test_data_{args.word_embedding}_og.pkl'
        loss_func = nn.CrossEntropyLoss()
        layers = [embedding_dim, 500, 5]
        drops = [0, 0]
        encoder = create_encoder(embedding_dim, args.encoder_model)
        classifier = DownstreamWrapper(args.model_name, args.saved_models, "sst_classification", train_data, test_data, encoder, layers, drops)
        opt_func = torch.optim.Adagrad(classifier.model.parameters(), lr=args.lr, weight_decay=args.wd)
        classifier.train(loss_func, opt_func, 15)
        classifier.save()
        print(classifier.test_accuracy())

    elif args.task.startswith("probe"):
        probing_task = args.task.split("_", 1)[1]
        train_data = f'./data/senteval_probing/{probing_task}_train_tree.pkl'
        test_data = f'./data/senteval_probing/{probing_task}_test_tree.pkl'
        loss_func = nn.CrossEntropyLoss()
        layers = [embedding_dim, 200, 6]
        drops = [0, 0]
        encoder = create_encoder(embedding_dim, args.encoder_model)
        wrapper = ProbingWrapper(args.model_name, args.saved_models, probing_task, train_data, test_data, encoder, layers, drops)
        opt_func = torch.optim.SGD(wrapper.model.parameters(), lr=args.lr, weight_decay=args.wd)
        wrapper.train(loss_func, opt_func, 10)
        wrapper.save()
        print(wrapper.test_accuracy())

    elif args.task == "data":
        we = load_glove('./data/glove.840B.300d.txt')
        nlp = spacy.load('en')
        dr = SentEvalDataReader('./data/senteval_probing/subj_number.txt')
        di = SentEvalDataIterator(dr, 'glove-wiki-gigaword-50', type_="tr", randomise=False)
        from random import shuffle
        train_data = di.all_data['tr']
        test_data = di.all_data['te']
        shuffle(train_data)
        shuffle(test_data)
        train_data_tree = []
        train_data_og = []
        for x in tqdm(train_data[:20000], total=20000):
            label = 0 if x[0] == "NNS" else 1
            sent_tree = tokenise_sent_tree(we, nlp, x[1].replace("\"", ""))
            sent_og = tokenise_sent_og(we, nlp, x[1].replace("\"", ""))
            train_data_tree.append((label, sent_tree))
            train_data_og.append((label, sent_og))
        pickle.dump(train_data_tree, Path('./data/senteval_probing/subj_number_train_tree.pkl').open('wb'))
        pickle.dump(train_data_og, Path('./data/senteval_probing/subj_number_train_og.pkl').open('wb'))
        test_data_tree = []
        test_data_og = []
        for x in tqdm(test_data[:5000], total=5000):
            label = 0 if x[0] == "NNS" else 1
            sent_tree = tokenise_sent_tree(we, nlp, x[1].replace("\"", ""))
            sent_og = tokenise_sent_og(we, nlp, x[1].replace("\"", ""))
            test_data_tree.append((label, sent_tree))
            test_data_og.append((label, sent_og))
        pickle.dump(test_data_tree, Path('./data/senteval_probing/subj_number_test_tree.pkl').open('wb'))
        pickle.dump(test_data_og, Path('./data/senteval_probing/subj_number_test_og.pkl').open('wb'))
