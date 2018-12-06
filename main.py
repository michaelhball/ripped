import argparse
import gensim.downloader as api
import math
import pickle
import spacy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from random import shuffle
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from modules.baseline_models import create_pooling_baseline
from modules.data_iterators import EasyIterator, SentEvalDataIterator, STSDataIterator
from modules.data_readers import SentEvalDataReader
from modules.models import create_encoder
from modules.model_wrappers import BaselineWrapper, ProbingWrapper, STSWrapper
from modules.utilities import all_dependencies, EmbeddingNode, my_dependencies, plot_train_test_loss, tokenise_sent, V


parser = argparse.ArgumentParser(description='PyTorch Dependency-Parse Encoding Model')
parser.add_argument('--word_embedding_source', type=str, default='glove-wiki-gigaword-50', help='word embedding source to use')
parser.add_argument('--word_embedding', type=str, default='glove_50', help='name of default word embeddings to use')
parser.add_argument('--saved_models', type=str, default='./models', help='directory to save/load models')
parser.add_argument('--encoder_model', type=str, default='3', help='id of desired encoder model')
parser.add_argument('--data_dir', type=str, default='./data', help='directory where data is stored')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--num_epochs', type=int, default=8, help='number of epochs to train for')
parser.add_argument('--model_name', type=str, default='sts_predictor', help='name of model to train')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
args = parser.parse_args()


def train_model(predictor, num_epochs, loss_func, opt_func, visualise=False, save=False):
    """
    Trains a model on STS dataset (optionally visualises loss and saves model).
    Args:
        predictor (BaseWrapper): model wrapper for model
        num_epochs (int): Number of training epochs
        loss_func (): loss function for training
        opt_func (): pytorch optimiser for training
        visualise (bool): indicator whether or not to plot loss
        save (bool): indicator whether or not to save the model
    Returns:
        None
    """
    train_losses, test_losses = predictor.train(loss_func, opt_func, num_epochs)
    if visualise:
        plot_train_test_loss(train_losses, test_losses, save_file=f'./data/sick/loss_plots/{args.model_name}.png')
    if save:
        predictor.save()


def evaluate_sentence(encoder, sentence):
    encoder.evaluate = True
    tree = tokenise_sent(sentence)
    print(tree)
    evaluated_tree = encoder(tree)
    print(evaluated_tree.representation)
    # print .representation at any level to see encoding for that
    # phrase


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


def nearest_neighbours(encoder, test_di, test_data, sent, k=10):
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

    we = api.load(args.word_embedding_source)
    nlp = spacy.load('en')
    enc = encoder(tokenise_sent(we, nlp, sent))
    neighbours = []
    for k, v in encodings.items():
        cosine_sim = F.cosine_similarity(enc, v).item()
        neighbours.append((k, cosine_sim))
    neighbours = sorted(neighbours, key=lambda x: x[1], reverse=True)

    return neighbours[:k]


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
        s1 = example[0]
        s2 = example[1]
        worst_predictions.append((x[1], x[2], s1, s2))
    
    return worst_predictions


if __name__ == "__main__":

    embedding_dim = int(args.word_embedding.split('_')[1])
    batch_size = 1
    sick_data = args.data_dir + '/sick'
    probing_data = args.data_dir + '/senteval_probing'

    # #######################################
    # # # param groups for ENC-3 model. USED FOR TRYING VARIOUS LR SCHEMES.
    # # pg1 = [{'params': predictor.model.encoder.lstm.parameters(), 'lr': lr}]
    # # pg2 = [{'params': ps, 'lr': lr} for ps in predictor.model.encoder.params.parameters()]
    # # param_groups = pg1 + pg2
    # # opt_func = opt(param_groups)
    # #######################################

    if args.task == "train" or args.task == "worst_predictions":
        train_di = EasyIterator(sick_data + f'/train_data_{args.word_embedding}.pkl')
        test_di = EasyIterator(sick_data + f'/test_data_{args.word_embedding}.pkl', randomise=False)
        weight_decay = 0
        layers = [50, 50, 1]
        drops = [0, 0]
        # predictor = STSWrapper(args.model_name, args.saved_models, embedding_dim, batch_size, my_dependencies, train_di, test_di, args.encoder_model, layers=layers, drops=drops)
        predictor = STSWrapper(args.model_name, args.saved_models, embedding_dim, batch_size, my_dependencies, train_di, test_di, args.encoder_model)

        if args.task == "worst_predictions":
            test_data = pickle.load(Path(sick_data+'/test.pkl').open('rb'))
            print(worst_predictions(predictor, test_data))
        else:
            loss_func = nn.MSELoss()
            opt_func = torch.optim.Adam(predictor.model.parameters(), lr=6e-4, weight_decay=0, amgsgrad=False)
            train_model(predictor, args.num_epochs, loss_func, opt_func, visualise=True, save=True)
            pearson, spearman, info = predictor.test_correlation()
            print(pearson, spearman)
        
    elif args.task == "nearest_neighbours" or args.task == "test_similarity":
        encoder = create_encoder(embedding_dim, batch_size, my_dependencies, args.encoder_model)
        path = args.saved_models+f'/{args.model_name}_encoder.pt'
        encoder.load_state_dict(torch.load(path))
        encoder.eval()

        if args.task == "nearest_neighbours":
            test_data = pickle.load(Path(sick_data+'/test.pkl').open('rb'))
            test_di = EasyIterator(sick_data+f'/test_data_{args.word_embedding}.pkl', randomise=False)
            sent = "A guy is mowing the lawn"
            print(nearest_neighbours(encoder, test_di, test_data, sent, k=10))
        else:
            print(test_similarity(encoder))

    elif args.task == "probing_model":
        train_nodes = pickle.load(Path('./data/sick/train_data_glove_50.pkl').open('rb'))
        train_data = pickle.load(Path('./data/sick/train.pkl').open('rb'))
        probe_model(predictor, train_data[60][:2], train_nodes[60][:2])

    elif args.task.startswith("probe"):
        probing_task = args.task.split("_", 1)[1]
        path = probing_data + f'/{probing_task}_'
        loss_func = nn.CrossEntropyLoss()
        opt = torch.optim.SGD
        lr = 0.01
        weight_decay = 5e-3
        num_classes = 6
        layers = [50, 200, num_classes]
        drops = [0, 0, 0]
        encoder = create_encoder(embedding_dim, batch_size, my_dependencies, args.encoder_model)
        wrapper = ProbingWrapper(args.model_name, args.saved_models, probing_task, path+'tr.pkl', path+'va.pkl', path+'te.pkl', encoder, layers, drops)
        opt_func = opt(wrapper.model.parameters(), lr=lr, weight_decay=weight_decay)
        wrapper.train(loss_func, opt_func, 5)
        wrapper.save()
        print(wrapper.test_accuracy())

    elif args.task == "test_baseline":
        train_data = pickle.load(Path(sick_data+f'/train_data_{args.word_embedding}_baseline.pkl').open('rb'))
        test_data = pickle.load(Path(sick_data+f'/test_data_{args.word_embedding}_baseline.pkl').open('rb'))
        loss_func = nn.MSELoss()
        opt = torch.optim.SGD
        lr = 0.01
        layers = [100, 250, 1]
        drops = [0, 0]
        pool_type = 'avg'
        predictor = BaselineWrapper(args.model_name, args.saved_models, train_data, test_data, layers, drops, pool_type)
        opt_func = opt(predictor.model.parameters(), lr=lr)
        predictor.train(loss_func, opt_func, 30)
        predictor.save()
        pearson, spearman, info = predictor.test_correlation()
        print(pearson, spearman)



    # IN FUTURE ONLY USE PARTIAL DATASETS FOR THESE PROBING TASKS - they're just too big to convert to EmbeddingNodes efficiently.
    # dr = SentEvalDataReader('./data/senteval_probing/sentence_length.txt')
    # di = SentEvalDataIterator(dr, 'glove-wiki-gigaword-50', type_="tr", randomise=False)
    # data = [[int(example[0]), di.tokenise_sent(example[1])] for example in di.data]
    # pickle.dump(data, Path('./data/senteval_probing/sentence_length_tr.pkl').open('wb'))
    # di.change_type("va")
    # data = [[int(example[0]), di.tokenise_sent(example[1])] for example in di.data]
    # pickle.dump(data, Path('./data/senteval_probing/sentence_length_va.pkl').open('wb'))

    # # past_present
    # dr = SentEvalDataReader('./data/senteval_probing/past_present.txt')
    # di = SentEvalDataIterator(dr, 'glove-wiki-gigaword-50', type_="tr", randomise=False)
    # data = [[0 if example[0] == "PRES" else 1, di.tokenise_sent(example[1].replace("\"", ""))] for example in di.data]
    # pickle.dump(data, Path('./data/senteval_probing/past_present_tr.pkl').open('wb'))
    # di.change_type("va")
    # data = [[0 if example[0] == "PRES" else 1, di.tokenise_sent(example[1].replace("\"", ""))] for example in di.data]
    # pickle.dump(data, Path('./data/senteval_probing/past_present_va.pkl').open('wb'))
    # di.change_type("te")
    # data = [[0 if example[0] == "PRES" else 1, di.tokenise_sent(example[1].replace("\"", ""))] for example in di.data]
    # pickle.dump(data, Path('./data/senteval_probing/past_present_te.pkl').open('wb'))