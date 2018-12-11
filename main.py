import argparse
import gensim.downloader as api
import pickle
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm

from modules.data_iterators import EasyIterator, SentEvalDataIterator
from modules.data_readers import SentEvalDataReader
from modules.models import create_encoder
from modules.model_wrappers import BaselineWrapper, DownstreamWrapper, ProbingWrapper, STSWrapper, TuningWrapper
from modules.utilities import all_dependencies, my_dependencies, plot_train_test_loss, tokenise_sent, universal_tags, V


parser = argparse.ArgumentParser(description='PyTorch Dependency-Parse Encoding Model')
parser.add_argument('--word_embedding_source', type=str, default='glove-wiki-gigaword-50', help='word embedding source to use')
parser.add_argument('--word_embedding', type=str, default='glove_50', help='name of default word embeddings to use')
parser.add_argument('--saved_models', type=str, default='./models', help='directory to save/load models')
parser.add_argument('--encoder_model', type=str, default='3', help='id of desired encoder model')
parser.add_argument('--data_dir', type=str, default='./data', help='directory where data is stored')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--model_name', type=str, default='sts_predictor', help='name of model to train')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
parser.add_argument('--baseline_type', type=str, default='pool', help='type of baseline model')
parser.add_argument('--pool_type', type=str, default='max', help='type of pooling for baseline (if pool baseline)')
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
    train_losses, test_losses = predictor.train(loss_func, opt_func, num_epochs=num_epochs)
    if visualise:
        plot_train_test_loss(train_losses, test_losses, save_file=f'./data/sick/loss_plots/{args.model_name}.png')
    if save:
        predictor.save()


def get_embedding_at_node(node, word):
    if node.text == word:
        return node.embedding
    for c in node.chidren:
        val = get_embedding_at_node(c, word)
        if not val is None:
            return val
    return None


def sim(e1, e2):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(e1, e2)[0][0]


def phrase_representation(encoder, we, nlp, phrase):
    return encoder(tokenise_sent(we, nlp, phrase)).representation


def doot(encoder, node, sentence_rep): # this gets nodes, not actual words - also useful though
    print(node.text, sim(sentence_rep, node.representation))
    for c in node.chidren:
        doot(encoder, c, sentence_rep)


def evaluate_sentence(encoder):
    encoder.evaluate = True
    we = api.load(args.word_embedding_source)
    nlp = spacy.load('en')


    # y = phrase_representation(encoder, we, nlp, "The dog was running in the park")
    # z = phrase_representation(encoder, we, nlp, "The dog was playing in the park")
    # a = phrase_representation(encoder, we, nlp, "running")
    # b = phrase_representation(encoder, we, nlp, "playing")
    # print(sim(y,z))
    # print(sim(y-a+b,z))
    # print(sim(z-b+a,y))

    # c = phrase_representation(encoder, we, nlp, "The child was running in the park")
    # d = phrase_representation(encoder, we, nlp, "The dog was running in the park")
    # e = phrase_representation(encoder, we, nlp, "The child")
    # f = phrase_representation(encoder, we, nlp, "The dog")
    # print(sim(c,d))
    # print(sim(c-e+f,d))
    # print(sim(d-f+e,c))

    # g = phrase_representation(encoder, we, nlp, "The dog was running in the field")
    # h = phrase_representation(encoder, we, nlp, "The dog was running in the sand")
    # i = phrase_representation(encoder, we, nlp, "in the field")
    # j = phrase_representation(encoder, we, nlp, "in the sand")
    # print(sim(g,h))
    # print(sim(g-i+j,h))
    # print(sim(h-j+i,g))

    k = phrase_representation(encoder, we, nlp, "A group of friends are riding the current in a raft")
    l = phrase_representation(encoder, we, nlp, "A group of friends are surfing the current in a raft")
    m = phrase_representation(encoder, we, nlp, "are riding")
    n = phrase_representation(encoder, we, nlp, "are surfing")
    # print(sim(k,l))
    # print(sim(k-m+n,l))
    # print(sim(l-n+m,k))

    ############################################
    # KEEP THIS HERE TO PUT IN MY RESULTS
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
    ############################################

    # tree = encoder(tokenise_sent(we, nlp, "A group of friends are riding the current in a raft"))
    # print(sim(k, get_embedding_at_node(tree, "are riding")))
    # tree = encoder(tokenise_sent(we, nlp, "A group of friends are surfing the current in a raft"))
    # print(sim(l, get_embedding_at_node(tree, "are surfing")))


    # sne(encoder)


def sne(encoder):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    encoder.evaluate = True
    we = api.load(args.word_embedding_source)
    nlp = spacy.load('en')
    sentences = {
        "d1": "dog",
        "d2": "dogs",
        "d3": "two dogs",
        "d4": "two beautiful dogs",
        "d5": "two dogs are playing",
        "d6": "two beautiful dogs are playing",
        "d7": "two dogs are playing on the beach",
        "d8": "two beautiful dogs are playing on the beach",
        "c1": "cat",
        "c2": "cats",
        "c3": "two cats",
        "c4": "two beautiful cats",
        "c5": "two cats are playing",
        "c6": "two beautiful cats are playing",
        "c7": "two cats are playing on the beach",
        "c8": "two beautiful cats are playing on the beach",
    }
    
    labels = []
    reprs = []
    for l, s in sentences.items():
        labels.append(l)
        reprs.append(phrase_representation(encoder, we, nlp, s)[0])
    
    print(reprs[0])
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(reprs)

    x, y = [], []
    for val in new_values:
        x.append(val[0])
        y.append(val[1])
    
    plt.figure(figsize=(10, 10))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
    plt.show()


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
    for sent, emb in encodings.items():
        cosine_sim = F.cosine_similarity(enc, emb).item()
        neighbours.append((sent, cosine_sim))
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


def bert_encoding(model, tokenizer, sentence):
    tokens = tokenizer.tokenize(sentence)
    indices = tokenizer.convert_tokens_to_ids(tokens)
    tensors = torch.tensor([indices])
    segments = torch.tensor([0 for i in range(len(indices))])
    _, pooled = model(tensors, segments)

    return pooled


def test_bert():
    from pytorch_pretrained_bert import BertTokenizer, BertModel
    from scipy.stats.stats import pearsonr, spearmanr
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')
    # model.eval()

    # loss_func = nn.MSELoss()
    # test_data = pickle.load(Path('./data/sick/test.pkl').open('rb'))
    # total_loss = 0.0
    # preds, scores, info = [], [], []
    # for i, (s1, s2, score) in tqdm(enumerate(test_data)):
    #     pooled1 = bert_encoding(model, tokenizer, s1)
    #     pooled2 = bert_encoding(model, tokenizer, s2)
    #     pred = F.cosine_similarity(pooled1, pooled2)
    #     total_loss += loss_func(pred, V((float(score)-1)/4)).item()
    #     preds.append(pred.item())
    #     scores.append((float(score)-1)/4)
    #     info.append((s1, s2, float(score), pred.item()))
    
    # avg_loss = total_loss / len(test_data)
    # print(avg_loss)
    
    # pickle.dump(preds, Path('./preds.pkl').open('wb'))
    # pickle.dump(scores, Path('./scores.pkl').open('wb'))
    # pickle.dump(info, Path('./info.pkl').open('wb'))

    preds = pickle.load(Path('./preds.pkl').open('rb'))
    scores = pickle.load(Path('./scores.pkl').open('rb'))
    info = pickle.load(Path('./info.pkl').open('rb'))

    pearson = pearsonr(preds, scores)
    spearman = spearmanr(preds, scores)
    avg_loss = 0

    return pearson, spearman, info, avg_loss


def x(score):
    if score < 0.2:
        return 0
    elif score < 0.4:
        return 1
    elif score < 0.6:
        return 2
    elif score < 0.8:
        return 3
    else:
        return 4


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

    if args.task == "train" or args.task == "worst_predictions" or args.task == "tune_similarity_predictor":
        # train_di = EasyIterator(sick_data + f'/train_data_{args.word_embedding}_trees.pkl')
        # test_di = EasyIterator(sick_data + f'/test_data_{args.word_embedding}_trees.pkl', randomise=False)
        layers = [2*embedding_dim, 250, 1]
        drops = [0, 0]
        # predictor = STSWrapper(args.model_name, args.saved_models, embedding_dim, batch_size, my_dependencies, train_di, test_di, args.encoder_model, layers=layers, drops=drops)
        # predictor = STSWrapper(args.model_name, args.saved_models, embedding_dim, batch_size, my_dependencies, train_di, test_di, args.encoder_model)

        train_di = EasyIterator(f'./data/sick/train_data_{args.word_embedding}_og.pkl')
        test_di = EasyIterator(f'./data/sick/test_data_{args.word_embedding}_og.pkl', randomise=False)
        predictor = STSWrapper(args.    model_name, args.saved_models, embedding_dim, batch_size, universal_tags, train_di, test_di, "pos", layers=layers, drops=drops)
        # predictor = STSWrapper(args.model_name, args.saved_models, embedding_dim, batch_size, universal_tags, train_di, test_di, "pos")

        if args.task == "train":
            loss_func = nn.MSELoss()
            opt_func = torch.optim.Adam(predictor.model.parameters(), lr=1e-4, weight_decay=0, amsgrad=False)
            # opt_func = torch.optim.Adagrad(predictor.model.parameters(), lr=0.01, weight_decay=0)
            # opt_func = torch.optim.SGD(predictor.model.parameters(), lr=0.01, weight_decay=0)
            train_model(predictor, args.num_epochs, loss_func, opt_func, visualise=True, save=True)
            pearson, spearman, info = predictor.test_correlation()
            print(pearson, spearman)

        if args.task == "worst_predictions":
            test_data = pickle.load(Path(sick_data+'/test.pkl').open('rb'))
            print(worst_predictions(predictor, test_data))
        
    elif args.task == "nearest_neighbours" or args.task == "test_similarity" or args.task == "evaluate_sentence":
        encoder = create_encoder(embedding_dim, batch_size, my_dependencies, args.encoder_model)
        path = args.saved_models+f'/{args.model_name}_encoder.pt'
        encoder.load_state_dict(torch.load(path))
        encoder.eval()

        if args.task == "nearest_neighbours":
            test_data = pickle.load(Path(sick_data+'/test.pkl').open('rb'))
            test_di = EasyIterator(sick_data+f'/test_data_{args.word_embedding}.pkl', randomise=False)
            sent = "A guy is mowing the lawn"
            print(nearest_neighbours(encoder, test_di, test_data, sent))
        elif args.task == "test_similarity":
            print(test_similarity(encoder))
        elif args.task == "evaluate_sentence":
            evaluate_sentence(encoder)
    
    elif args.task == "test_classification":
        train_data = f'./data/sst/train_data_{args.word_embedding}_binary.pkl'
        test_data = f'./data/sst/test_data_{args.word_embedding}_binary.pkl'
        loss_func = nn.CrossEntropyLoss()
        opt = torch.optim.Adam
        lr = 0.001
        weight_decay = 0
        num_classes = 2
        layers = [50, 250, num_classes]
        drops = [0, 0]
        encoder = create_encoder(embedding_dim, batch_size, my_dependencies, args.encoder_model)
        wrapper = DownstreamWrapper(args.model_name, args.saved_models, "sst_classification", train_data, test_data, encoder, layers, drops)
        opt_func = opt(wrapper.model.parameters(), lr=lr, weight_decay=weight_decay)
        wrapper.train(loss_func, opt_func, 10)
        wrapper.save()
        print(wrapper.test_accuracy())

    elif args.task.startswith("probe"):
        probing_task = args.task.split("_", 1)[1]
        path = probing_data + f'/{probing_task}_'
        loss_func = nn.CrossEntropyLoss()
        opt = torch.optim.SGD
        lr = 0.01
        weight_decay = 5e-3
        num_classes = 2
        layers = [50, 200, num_classes]
        drops = [0, 0, 0]
        encoder = create_encoder(embedding_dim, batch_size, my_dependencies, args.encoder_model)
        wrapper = ProbingWrapper(args.model_name, args.saved_models, probing_task, path+'tr.pkl', path+'va.pkl', path+'te.pkl', encoder, layers, drops)
        opt_func = opt(wrapper.model.parameters(), lr=lr, weight_decay=weight_decay)
        wrapper.train(loss_func, opt_func, 5)
        wrapper.save()
        print(wrapper.test_accuracy())

    elif args.task == "test_baseline":
        train_data = sick_data + f'/train_data_{args.word_embedding}_baseline.pkl'
        test_data = sick_data + f'/test_data_{args.word_embedding}_baseline.pkl'
        loss_func = nn.MSELoss()
        layers = [2*embedding_dim, 250, 1]
        drops = [0, 0]
        if args.baseline_type == "pool":
            predictor = BaselineWrapper(args.model_name, args.saved_models, train_data, test_data, layers, drops, args.baseline_type, args.pool_type)
            opt_func = torch.optim.SGD(predictor.model.parameters(), lr=0.01)
        elif args.baseline_type.startswith("lstm"):
            predictor = BaselineWrapper(args.model_name, args.saved_models, train_data, test_data, layers, drops, args.baseline_type, embedding_dim=embedding_dim, num_layers=1)
            opt_func = torch.optim.Adam(predictor.model.parameters(), lr=0.01)
        predictor.train(loss_func, opt_func, 15)
        predictor.save()
        pearson, spearman, info = predictor.test_correlation()
        print(pearson, spearman)
    
    elif args.task == "test_sota":
        p, s, info, avg_loss = test_bert()
        print(p, s, avg_loss)

    elif args.task == "data":
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

        # # # bigram shift
        # dr = SentEvalDataReader('./data/senteval_probing/bigram_shift.txt')
        # di = SentEvalDataIterator(dr, 'glove-wiki-gigaword-50', type_="tr", randomise=False)
        # data = [[0 if example[0] == "I" else 1, di.tokenise_sent(example[1].replace("\"", ""))] for example in di.data]
        # pickle.dump(data, Path('./data/senteval_probing/bigram_shift_tr.pkl').open('wb'))
        # di.change_type("va")
        # data = [[0 if example[0] == "I" else 1, di.tokenise_sent(example[1].replace("\"", ""))] for example in di.data]
        # pickle.dump(data, Path('./data/senteval_probing/bigram_shift_va.pkl').open('wb'))
        # di.change_type("te")
        # data = [[0 if example[0] == "I" else 1, di.tokenise_sent(example[1].replace("\"", ""))] for example in di.data]
        # pickle.dump(data, Path('./data/senteval_probing/bigram_shift_te.pkl').open('wb'))

        # we = api.load(args.word_embedding_source)
        # nlp = spacy.load('en')
        # train_data = pickle.load(Path('./data/sst/train_data.pkl').open('rb'))
        # test_data = pickle.load(Path('./data/sst/test_data.pkl').open('rb'))
        # train_data = [(tokenise_sent(we, nlp, x[0].replace('\"', "")), x[1]) for x in train_data]
        # print('done train')
        # test_data = [(tokenise_sent(we, nlp, x[0].replace('\"', "")), x[1]) for x in test_data]
        # pickle.dump(train_data, Path('./data/sst/train_data_glove_50.pkl').open('wb'))
        # pickle.dump(test_data, Path('./data/sst/test_data_glove_50.pkl').open('wb'))

        # from modules.utilities import EmbeddingNode
        # train = pickle.load(Path('./data/sick/train.pkl').open('rb'))
        # test = pickle.load(Path('./data/sick/test.pkl').open('rb'))
        # nlp = spacy.load('en')
        # we = api.load('fasttext-wiki-news-subwords-300')
        # new_train, new_test = [], []
        # for x in tqdm(iterable=train, total=len(train)):
        #     new_node1 = EmbeddingNode(we, list(nlp(str(x[0])).sents)[0].root)
        #     new_node2 = EmbeddingNode(we, list(nlp(str(x[1])).sents)[0].root)
        #     new_train.append((new_node1, new_node2, float(x[2])))
        # for x in tqdm(iterable=test, total=len(test)):
        #     new_node1 = EmbeddingNode(we, list(nlp(str(x[0])).sents)[0].root)
        #     new_node2 = EmbeddingNode(we, list(nlp(str(x[1])).sents)[0].root)
        #     new_test.append((new_node1, new_node2, float(x[2])))
        # pickle.dump(new_train, Path('./data/sick/train_data_fasttext_300_pos.pkl').open('wb'))
        # pickle.dump(new_test, Path('./data/sick/test_data_fasttext_300_pos.pkl').open('wb'))
            
        # import numpy as np
        # nlp = spacy.load('en')
        # we = {}
        # with Path('./data/glove.840b.300d.txt').open('r') as f:
        #     for line in tqdm(f, total=2196018):
        #         l = line.split(' ')
        #         we[l[0]] = np.asarray(l[1:], dtype=np.float32)
        # print(f'loaded word embeddings, #={len(we)}')
        # train = pickle.load(Path('./data/sick/train.pkl').open('rb'))
        # test = pickle.load(Path('./data/sick/test.pkl').open('rb'))
        # new_train, new_test = [], []
        # for x in tqdm(iterable=train, total=len(train)):
        #     s1 = [(t.pos_, we[t.text] if t.text in we else we['unk']) for t in nlp(str(x[0]))]
        #     s2 = [(t.pos_, we[t.text] if t.text in we else we['unk']) for t in nlp(str(x[1]))]
        #     new_train.append((s1,s2,float(x[2])))
        # print(new_train[0])
        # pickle.dump(new_train, Path('./data/sick/train_data_glove_300_og.pkl').open('wb'))
        # for x in tqdm(iterable=test, total=len(test)):
        #     s1 = [(t.pos_, we[t.text] if t.text in we else we['unk']) for t in nlp(str(x[0]))]
        #     s2 = [(t.pos_, we[t.text] if t.text in we else we['unk']) for t in nlp(str(x[1]))]
        #     new_test.append((s1,s2,float(x[2])))
        # print(new_test[0])
        # pickle.dump(new_test, Path('./data/sick/test_data_glove_300_og.pkl').open('wb'))




        None
