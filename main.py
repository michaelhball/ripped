import argparse
import gensim.downloader as api
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import spacy
import time
import torch
import torch.nn as nn
import torch.nn.functional as Fs

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid
from torchtext import data, vocab
from tqdm import tqdm

from modules.data_iterators import *
from modules.data_readers import IntentClassificationDataReader, SentEvalDataReader
from modules.models import create_encoder
from modules.model_wrappers import BaselineWrapper, DownstreamWrapper, IntentWrapper, MTLWrapper, ProbingWrapper, STSWrapper
from modules.utilities import *


parser = argparse.ArgumentParser(description='PyTorch Dependency-Parse Encoding Model')
parser.add_argument('--word_embedding_source', type=str, default='glove-wiki-gigaword-50', help='word embedding source to use')
parser.add_argument('--word_embedding', type=str, default='glove_50', help='name of default word embeddings to use')
parser.add_argument('--saved_models', type=str, default='./models', help='directory to save/load models')
parser.add_argument('--encoder_model', type=str, default='pos_tree', help='pos_lin/pos_tree/dep_tree')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--model_name', type=str, default='999', help='name of model to train')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
parser.add_argument('--subtask', type=str, default='none', help='sub-task within whichever specified task')
parser.add_argument('--baseline_type', type=str, default='pool', help='type of baseline model')
parser.add_argument('--lr', type=float, default=6e-4, help='learning rate for whichever model is being trained')
parser.add_argument('--wd', type=float, default=0, help='L2 regularization for training')
parser.add_argument('--frac', type=float, default=1, help='fraction of training data to use')
args = parser.parse_args()


def evaluate_sentences(encoder):
    encoder.evaluate = True
    we = api.load(args.word_embedding_source)
    nlp = spacy.load('en')
    o = phrase_representation(encoder, we, nlp, "Some penguins are riding the current in a raft")
    p = phrase_representation(encoder, we, nlp, "A group of dogs are riding the human as a ship")
    pp = phrase_representation(encoder, we, nlp, "A group of dogs are riding the human as a mule")
    print(sim(o,p))
    print(sim(o,pp))
    q = phrase_representation(encoder, we, nlp, "in a raft")
    r = phrase_representation(encoder, we, nlp, "as a ship")
    s = phrase_representation(encoder, we, nlp, "as a mule")
    t = phrase_representation(encoder, we, nlp, "Some penguins")
    u = phrase_representation(encoder, we, nlp, "A group of dogs")
    print(sim(q,r))
    print(sim(q,s))
    print(sim(t,u))

    def phrase_representation(encoder, we, nlp, phrase):
        return encoder(tokenise_sent(we, nlp, phrase)).representation

    def sim(e1, e2):
        return cosine_similarity(e1, e2)[0][0]


def test_similarity(encoder):
    """
    Tests the similarity of a base sentence to any number
        of others.
    Args:
        encoder (DependencyEncoder): a pretrained, loaded encoder
    Returns:
        The similarity scores for the compared sentences.
    """
    we = api.load(args.word_embedding_source)
    nlp = spacy.load('en')

    s1 = "A waterfall is flowing out of a shallow pool"
    sentences = [
        "Out of a shallow pool, a waterfall is flowing"
    ]

    enc1 = encoder(tokenise_sent(we, nlp, s1))
    sims = []
    for s in sentences:
        enc2 = encoder(tokenise_sent(we, nlp, s))
        sims.append((s, round(F.cosine_similarity(enc1, enc2).item(), 3)))

    return sims


def nearest_neighbours(encoder, test_di, test_data, sentences, k=20):
    """
    Gets the nearest neighbours from the SICK test dataset to 
        a given sentence (not necessarily in dataset).
    Args:
        Encoder (DependencyEncoder): a loaded encoder
        test_di (DataIterator): data as EmbeddingNodes
        test_data (list): Unformatted test data
        sent (str): the considered sentence
        k (int): Number of neighbours to return
    Returns:
        List of the k nearest neighbours to given sentence.
    """
    encodings = {}
    for i, x in enumerate(iter(test_di)):
        s1 = test_data[i][0]
        s2 = test_data[i][1]
        encodings[s1] = encoder(x[0])
        encodings[s2] = encoder(x[1])

    if args.word_embedding == ('glove_300'):
        from modules.utilities import load_glove
        we = load_glove('./data/glove.840B.300d.txt')
    else:
        we = api.load(args.word_embedding_source)

    nlp = spacy.load('en')
    sent_neighbours = []
    for s in sentences:
        if args.encoder_model == "pos_lin":
            enc = encoder(tokenise_sent_og(we, nlp, s))
        else:
            enc = encoder(tokenise_sent_tree(we, nlp, s))
        neighbours = []
        for sent, emb in encodings.items():
            # cosine_sim = F.cosine_similarity(enc, emb).item()
            cosine_sim = cosine_similarity(enc, emb)[0][0]
            neighbours.append((sent, round(cosine_sim,3)))
        print('\n'+s)
        print(sorted(neighbours, key=lambda x: x[1], reverse=True)[:k])


def worst_predictions(predictor, test_data, k=10):
    """
    Calculates the worst predictions by a given model.
    Args:
        predictor (BaseWrapper): A model wrapper
        test_data (list): Unformatted test data to access strings.
    Returns a list of the k worst predictions.
    """
    pearson, spearman, info = predictor.test_correlation(load=True)
    worst_predictions = []
    for x in info[:k]:
        example = test_data[x[0]]
        s1, s2 = example[0], example[1]
        worst_predictions.append((x[1], x[2], s1, s2))

    print(worst_predictions)


def get_node(node, word, dep):
    if node.text == word and node.dep == dep:
        return node
    for c in node.chidren:
        n = get_node(c, word, dep)
        if not n is None:
            return n
    return None


def create_plot(save_file, title, spans, percentages):
    fig = plt.figure()
    plt.xticks(np.arange(len(spans)), spans, rotation=0)
    plt.bar(spans, percentages)
    plt.ylabel('%')
    plt.title(title)
    plt.tight_layout()
    fig.savefig(save_file)


def get_token_span(token):
    lefts, rights = list(token.lefts), list(token.rights)
    leftmost_idx, rightmost_idx = token.i, token.i
    while lefts:
        l = lefts.pop()
        if l.i < leftmost_idx:
            leftmost_idx = l.i
        lefts.extend(list(l.lefts))
    while rights:
        l = rights.pop()
        if l.i > rightmost_idx:
            rightmost_idx = l.i
        rights.extend(list(l.rights))

    return leftmost_idx, rightmost_idx


def create_pos_lin_visualisations(save_file, sentence, sentence_embeddings, encoder):
    nlp = spacy.load('en')
    sent = list(nlp(str(sentence).lower()).sents)[0]
    sentence_encoding, word_embeddings = encoder(sentence_embeddings)
    nonzeros = sentence_encoding[0].nonzero()[0]
    tokens, percentages = [], []
    for we, t in zip(word_embeddings, sent):
        tokens.append(f'{t.text}\n{t.i}')
        percentages.append((sentence_encoding[0][nonzeros]==we[0][nonzeros]).sum() / float(len(nonzeros)))
    
    create_plot(f"{save_file}_overall.png", f"\"{str(sent)}\"", tokens, percentages)


def create_pos_tree_visualisations(save_file, sentence, sentence_tree, encoder):
    nlp = spacy.load('en')
    sent = list(nlp(str(sentence).lower()).sents)[0]
    encoded_tree = encoder(sentence_tree)

    for token in sent:
        if list(token.children):
            l,r = get_token_span(token)
            token_span = str(sent[l:r+1])
            node = get_node(encoded_tree, token.text, token.dep_)
            representation = node.representation[0]
            nonzeros = representation.nonzero()[0]
            spans, percentages, span_indices = [], [], []
            
            for child_token in token.children:
                child_span_left, child_span_right = get_token_span(child_token)
                spans.append(f'{str(sent[child_span_left:child_span_right+1])}\n{child_span_left}:{child_span_right}')
                span_indices.append((child_span_left, child_span_right))
                child_node = get_node(encoded_tree, child_token.text, child_token.dep_)
                percentages.append((representation[nonzeros] == child_node.representation[0][nonzeros]).sum() / float(len(nonzeros)))
            
            n_list = [n for n, i in enumerate(span_indices) if token.i > i[1]]
            n = n_list[-1]+1 if n_list else 1
            spans.insert(n, f'{token.text}\n{token.i}')
            percentages.insert(n, (representation[nonzeros] == node.embedding[0][nonzeros]).sum() / float(len(nonzeros)))
            # plots[token.i] = create_plot(f"relative importance of dependents to '{token.text}' encoding", spans, percentages)
            create_plot(f'{save_file}_{token.i}.png', f"relative importance of dependents to '{token.text}' encoding", spans, percentages)

    percentage_map = {}
    encoding = encoded_tree.representation[0]
    nonzeros = encoding.nonzero()[0]
    next_children = [encoded_tree]
    while next_children:
        c = next_children.pop(0)
        percentage_map[c.text] = (encoding[nonzeros] == c.embedding[0][nonzeros]).sum() / float(len(nonzeros))
        next_children += c.chidren
    tokens = [f'{t.text}\n{t.i}' for t in sent]
    percentages = [percentage_map[t.text] for t in list(sent)]
    create_plot(f"{save_file}_overall.png", f"\"{str(sent)}\"", tokens, percentages)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def grid_search(saved_models, train_ds, val_ds, test_ds, param_grid, encoder_type, layers, TEXT, LABEL, k=10, verbose=True):
    """
    Perform grid search over given parameter options.
    Args:
        saved_models (str): directory to save temporary models
        train_ds (Dataset): torchtext dataset with training data
        val_ds (Dataset): torchtext dataset with validation data
        test_ds (Dataset): torchtext dataset with testing data
        param_grid (dict): param values: {str: list(values)}
        encoder_type (str): encoder type to use
        layers (list(int)): list of feedforward layers for classifier
        TEXT (Field): torchtext field representing x values
        LABEL (LabelField): torchtext labelfield representing y values
        k (int): number of training runs to perform for each trial
    Returns:
        List of trials sorted by mean accuracy (each trial performs k training runs)
    """
    results = []
    for i, params in enumerate(ParameterGrid(param_grid)):
        lr = params['lr']
        bs = params['bs']
        drops = [params['drop1'], params['drop2']]
        loss_func = nn.CrossEntropyLoss()
        mean, std = repeat_trainer(saved_models, loss_func, train_ds, val_ds, test_ds, TEXT, LABEL, bs, layers, drops, lr, k=k, verbose=verbose)
        print("-------------------------------------------------------------------------------------------------------------------------------------------")
        print(i, params, mean, std)
        print("-------------------------------------------------------------------------------------------------------------------------------------------")
        results.append({'mean': mean, 'std': std, 'params': params})

    return(sorted(results, key=lambda x: x['mean']))


def repeat_trainer(saved_models, loss_func, train_ds, val_ds, test_ds, text, label, bs, layers, drops, lr, frac=1, k=10, verbose=True):
    """
    Function to perform multiple training runs (for model validation).
    Args:
        saved_models (str): directory to save temporary models
        loss_func (): pytorch loss function for training
        train_ds (Dataset): torchtext dataset with training data
        val_ds (Dataset): torchtext dataset with validation data
        test_ds (Dataset): torchtext dataset with testing data
        text (Field): torchtext field representing x values
        label (LabelField): torchtext labelfield representing y values
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
    
    for i in range(k):
        # TO DO: Ideally i want to shuffle the training data each iteration here, before taking a fraction or creating iterators
        new_train_ds = train_ds
        name = args.model_name + f'_{i}'
        if frac != 1:
            examples = train_ds.examples
            np.random.shuffle(examples)
            new_train_ds = data.Dataset(examples[:int(len(examples)*frac)], {'x': text, 'y': label})
        train_di, val_di, test_di = get_intent_classification_data_iterators(new_train_ds, val_ds, test_ds, (bs,bs,bs))
        wrapper = IntentWrapper(name, saved_models, 300, text.vocab, args.encoder_model, train_di, val_di, test_di, layers=layers, drops=drops)
        opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=lr, betas=(0.7, 0.999), weight_decay=args.wd)
        train_losses, test_losses = wrapper.train(loss_func, opt_func, verbose=False)

    if verbose:
        elapsed_time = time.time() - start_time
        print(f"{k} training runs completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    
    accuracies = []
    for i in range(k):
        name = args.model_name + f'_{i}'
        train_di, val_di, test_di = get_intent_classification_data_iterators(train_ds, val_ds, test_ds, (bs,bs,bs))
        wrapper = IntentWrapper(name, saved_models, 300, text.vocab, args.encoder_model, train_di, val_di, test_di, layers=layers, drops=drops)
        accuracies.append(wrapper.test_accuracy(load=True))
    
    return np.mean(accuracies), np.std(accuracies)


if __name__ == "__main__":
    embedding_dim = int(args.word_embedding.split('_')[1])
    layers = [2*embedding_dim, 250, 1]
    drops = [0, 0]

    sick_train_data = './data/sts/sick/train_data'
    sick_test_data = './data/sts/sick/test_data'
    train_data_raw = pickle.load(Path('./data/sts/sick/train.pkl').open('rb'))
    test_data_raw = pickle.load(Path('./data/sts/sick/test.pkl').open('rb'))
    di_suffix = {"pos_lin": "og", "pos_tree": "trees", "dep_tree": "trees"}
    # train_di = EasyIterator(f'{sick_train_data}_{args.word_embedding}_{di_suffix[args.encoder_model]}.pkl')
    # test_di = EasyIterator(f'{sick_test_data}_{args.word_embedding}_{di_suffix[args.encoder_model]}.pkl', randomise=False)

    if args.task == "train_sts":
        predictor = STSWrapper(args.model_name, args.saved_models, embedding_dim, train_di, test_di, args.encoder_model, layers=layers, drops=drops)
        loss_func = nn.MSELoss()
        opt_func = torch.optim.Adam(predictor.model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
        train_losses, test_losses = predictor.train(loss_func, opt_func, args.num_epochs)
        # plot_train_test_loss(train_losses, test_losses, save_file=f'./data/sick/loss_plots/{args.model_name}.png')


    elif args.task == "chat_ic":
        intents = pickle.load(Path('./data/chat_ic/intents.pkl').open('rb'))
        C = len(intents)
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        BS = 128
        FRAC = 1
        layers = [300, 100, C]
        drops = [0, 0]

        train_ds, val_ds, test_ds = IntentClassificationDataReader('./data/chat_ic/', '_tknsd.pkl', TEXT, LABEL).read()
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings) # 461

        if args.subtask == "grid_search":
            param_grid = {'lr': [3e-3, 6e-3, 1e-2], 'drop1': [0, 0.1, 0.2], 'drop2': [0, 0.1, 0.2], 'bs': [128]}
            results = grid_search(f'{args.saved_models}/chat_ic', train_ds, val_ds, test_ds, param_grid, args.encoder_model, layers, TEXT, LABEL, k=5, verbose=False)
            print(results)
        elif args.subtask == "repeat_train":
            loss_func = nn.CrossEntropyLoss()
            mean, std = repeat_trainer(f'{args.saved_models}/{args.task}', loss_func, train_ds, val_ds, test_ds, TEXT, LABEL, BS, layers, drops, args.lr, verbose=True)
            print(mean, std)
        elif args.subtask == "train":
            train_di, val_di, test_di = get_intent_classification_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
            loss_func = nn.CrossEntropyLoss()
            wrapper = IntentWrapper(args.model_name, f'{args.saved_models}/chat_ic', 300, TEXT.vocab, args.encoder_model, train_di, val_di, test_di, layers=layers, drops=drops)
            opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=args.lr, betas=(0.7,0.999), weight_decay=args.wd)
            train_losses, val_losses = wrapper.train(loss_func, opt_func)
        

    elif args.task == "snipsnlu":
        # accuracies = [0.951,0.941,0.865,0.81,0.606,0.245]
        # stds = [0.002,0.015,0.02,0.036,0.168,0.195]
        # fracs = [1,0.1,0.01,0.005,0.002,0.001]
        # cis = [0.001146009,0.008595071,0.011460094,0.02062817,0.096264792,0.111735919]
        # plt.plot(fracs, accuracies, color='black')
        # plt.xlabel('fraction of training data'); plt.ylabel('accuracy')
        # plt.xscale('log')
        # plt.title('Accuracy using different fractions of training data')
        # plt.errorbar(fracs,accuracies,yerr=cis, ecolor='black',elinewidth=0.5,capsize=1)
        # plt.show()
        # assert(False)

        intents = ['add_to_playlist', 'book_restaurant', 'get_weather', 'play_music', 'rate_book', 'search_creative_work', 'search_screening_event']
        C = len(intents)

        # variables
        TEXT = data.Field(sequential=True)
        LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
        BS = 128
        FRAC = 0.002
        layers = [300, 100, C]
        drops = [0, 0]

        # get datasets & create vocab
        train_ds, val_ds, test_ds = IntentClassificationDataReader('./data/snipsnlu/', '_tknsd.pkl', TEXT, LABEL).read()
        glove_embeddings = vocab.Vectors("glove.840B.300d.txt", './data/')
        TEXT.build_vocab(train_ds, vectors=glove_embeddings) # 10196

        if args.subtask == "grid_search":
            param_grid = {'lr': [1e-3, 6e-4, 3e-3, 6e-3], 'drop1': [0, 0.1], 'drop2': [0, 0.1], 'bs': [128, 64]}
            ic_grid_search(train_ds, val_ds, test_ds, param_grid, args.encoder_model, layers, TEXT, LABEL)
        
        elif args.subtask == "repeat_train":
            loss_func = nn.CrossEntropyLoss()
            print(repeat_trainer(loss_func, train_ds, val_ds, test_ds, TEXT, LABEL, BS, layers, drops, args.lr, FRAC, k=10))

        else:
            train_di, val_di, test_di = get_intent_classification_data_iterators(train_ds, val_ds, test_ds, (BS,BS,BS))
            wrapper = IntentWrapper(args.model_name, f'{args.saved_models}/intent_class', 300, TEXT.vocab, args.encoder_model, train_di, val_di, test_di, layers=layers, drops=drops)
            loss_func = nn.CrossEntropyLoss()
            opt_func = torch.optim.Adam(wrapper.model.parameters(), lr=args.lr, betas=(0.7, 0.999), weight_decay=args.wd, amsgrad=False)
            train_losses, test_losses = wrapper.train(loss_func, opt_func)
            # print(wrapper.repeat_trainer(loss_func, torch.optim.Adam, args.lr, (0.7,0.999), args.wd, k=20))



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
