import argparse
import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import pickle
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as Fs

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from modules.data_iterators import EasyIterator, SentEvalDataIterator
from modules.data_readers import SentEvalDataReader
from modules.models import create_encoder
from modules.model_wrappers import BaselineWrapper, DownstreamWrapper, ProbingWrapper, STSWrapper
from modules.utilities import *


parser = argparse.ArgumentParser(description='PyTorch Dependency-Parse Encoding Model')
parser.add_argument('--word_embedding_source', type=str, default='glove-wiki-gigaword-50', help='word embedding source to use')
parser.add_argument('--word_embedding', type=str, default='glove_50', help='name of default word embeddings to use')
parser.add_argument('--saved_models', type=str, default='./models', help='directory to save/load models')
parser.add_argument('--encoder_model', type=str, default='pos_tree', help='pos_lin/pos_tree/dep_tree')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--model_name', type=str, default='999', help='name of model to train')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
parser.add_argument('--baseline_type', type=str, default='pool', help='type of baseline model')
parser.add_argument('--lr', type=float, default=6e-4, help='learning rate for whichever model is being trained')
parser.add_argument('--wd', type=float, default=0, help='L2 regularization for training')
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


if __name__ == "__main__":
    embedding_dim = int(args.word_embedding.split('_')[1])
    layers = [2*embedding_dim, 250, 1]
    drops = [0, 0]

    sick_train_data = './data/sick/train_data'
    sick_test_data = './data/sick/test_data'
    train_data_raw = pickle.load(Path('./data/sick/train.pkl').open('rb'))
    test_data_raw = pickle.load(Path('./data/sick/test.pkl').open('rb'))
    di_suffix = {"pos_lin": "og", "pos_tree": "trees", "dep_tree": "trees"}
    train_di = EasyIterator(f'{sick_train_data}_{args.word_embedding}_{di_suffix[args.encoder_model]}.pkl')
    test_di = EasyIterator(f'{sick_test_data}_{args.word_embedding}_{di_suffix[args.encoder_model]}.pkl', randomise=False)

    if args.task == "train":
        predictor = STSWrapper(args.model_name, args.saved_models, embedding_dim, train_di, test_di, args.encoder_model, layers=layers, drops=drops)
        loss_func = nn.MSELoss()
        opt_func = torch.optim.Adam(predictor.model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
        train_losses, test_losses = predictor.train(loss_func, opt_func, args.num_epochs)
        # plot_train_test_loss(train_losses, test_losses, save_file=f'./data/sick/loss_plots/{args.model_name}.png')

    elif args.task == "train_baseline":
        loss_func = nn.MSELoss()
        train_data = f'{sick_train_data}_{args.word_embedding}_baseline.pkl'
        test_data = f'{sick_test_data}_{args.word_embedding}_baseline.pkl'
        predictor = BaselineWrapper(args.model_name, args.saved_models, train_data, test_data, layers, drops, args.baseline_type, embedding_dim=embedding_dim)
        opt_func = torch.optim.Adam(predictor.model.parameters(), lr=args.lr)
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
        train_data = f'./data/senteval_probing/{probing_task}_train_og.pkl'
        test_data = f'./data/senteval_probing/{probing_task}_test_og.pkl'
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
        dr = SentEvalDataReader('./data/senteval_probing/sentence_length.txt')
        di = SentEvalDataIterator(dr, 'glove-wiki-gigaword-50', type_="tr", randomise=False)
        from random import shuffle
        train_data = di.all_data['tr']
        test_data = di.all_data['te']
        shuffle(train_data)
        shuffle(test_data)
        train_data_tree = []
        train_data_og = []
        for x in tqdm(train_data[:20000], total=20000):
            sent_tree = tokenise_sent_tree(we, nlp, x[1].replace("\"", ""))
            sent_og = tokenise_sent_og(we, nlp, x[1].replace("\"", ""))
            train_data_tree.append((int(x[0]), sent_tree))
            train_data_og.append((int(x[0]), sent_og))
        pickle.dump(train_data_tree, Path('./data/senteval_probing/sentence_length_train_tree.pkl').open('wb'))
        pickle.dump(train_data_og, Path('./data/senteval_probing/sentence_length_train_og.pkl').open('wb'))
        test_data_tree = []
        test_data_og = []
        for x in tqdm(test_data[:5000], total=5000):
            label = 0 if x[0] == "PRES" else 1
            sent_tree = tokenise_sent_tree(we, nlp, x[1].replace("\"", ""))
            sent_og = tokenise_sent_og(we, nlp, x[1].replace("\"", ""))
            test_data_tree.append((int(x[0]), sent_tree))
            test_data_og.append((int(x[0]), sent_og))
        pickle.dump(test_data_tree, Path('./data/senteval_probing/sentence_length_test_tree.pkl').open('wb'))
        pickle.dump(test_data_og, Path('./data/senteval_probing/sentence_length_test_og.pkl').open('wb'))
