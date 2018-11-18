import argparse
import gensim.downloader as api
import spacy
import time
import torch
import torch.nn as nn

from pathlib import Path
from scipy.stats.stats import pearsonr

from modules.data_iterators import STSDataIterator
from modules.models import create_sts_predictor, create_sentence_distance_sts_predictor
from modules.utilities import my_dependencies, V


parser = argparse.ArgumentParser(description='PyTorch Dependency-Parse Encoding Model')
parser.add_argument('--word_embedding_source', type=str, default='glove-wiki-gigaword-50', help='word embedding source to use')
parser.add_argument('--embedding_dim', type=int, default=50, help='size of word embeddings used')
parser.add_argument('--saved_models', type=str, default='./models', help='directory to save/load models')
parser.add_argument('--data_dir', type=str, default='./data', help='directory where data is stored')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--model_name', type=str, default='sts_predictor', help='name of model to train')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
args = parser.parse_args()


class STS():
    """
    A class for training an STS predictor.
    """
    def __init__(self, name, saved_models, word_embeddings, embedding_dim, train_di=None, dev_di=None, test_di=None):
        self.name = name
        self.saved_models = saved_models
        self.word_embeddings = word_embeddings
        self.embedding_dim = embedding_dim
        self.train_di, self.dev_di, self.test_di = train_di, dev_di, test_di
        self.nlp = spacy.load('en')
        if self.train_di: 
            self.num_train_examples = self.train_di.num_examples
        if self.dev_di:
            self.num_dev_examples = self.dev_di.num_examples
        if self.test_di:
            self.num_test_examples = self.test_di.num_examples

    def save(self):
        self._save_model()
        self._save_encoder()

    def _save_model(self):
        path = self.saved_models + '/{0}.pt'.format(self.name)
        torch.save(self.model.state_dict(), path)

    def _save_encoder(self):
        path = self.saved_models + '/{0}_encoder.pt'.format(self.name)
        torch.save(self.model.encoder.state_dict(), path)
    
    def _load_model(self, model_name):
        path = self.saved_models + '/{0}.pt'.format(model_name)
        self.model.load_state_dict(torch.load(path))

    def _load_encoder(self, encoder_name):
        path = self.saved_models + '/{0}_encoder.pt'.format(encoder_name)
        self.model.encoder.load_state_dict(torch.load(path))
    
    def _setup_model(self, batch_size):
        layers = [200, 50, 1] # totally random at this point
        self.model = create_sts_predictor(self.nlp, self.word_embeddings, self.embedding_dim, batch_size, my_dependencies, layers)

    def all_similarity(self):
        # self._setup_model(1)
        # self._load_model(self.name)
        self.model.eval()
        preds, scores, info = [], [], []
        for i, example in enumerate(iter(self.dev_di)):
            s1, s2, score = str(example[0][0]), str(example[0][1]), float(example[0][2])
            pred, relu = self.model(s1, s2)
            preds.append(pred.item())
            scores.append(score)
            info.append((s1, s2, score, pred.item()))
        
        pearson_coefficient = pearsonr(preds, scores)
        info = sorted(info, key=lambda tup: abs(tup[2]-tup[3]), reverse=True)
        
        return pearson_coefficient, info

    def similarity(self, input1, input2):
        self._setup_model(1)
        self._load_model(self.name)
        self.model.eval()
        pred, relu = self.model(input1, input)

        return pred.item()
    
    def _train(self, loss_func, opt_func, num_epochs):
        self.model.train()
        for e in range(num_epochs):
            total_loss = 0.0
            for i, example in enumerate(iter(self.train_di)):
                s1, s2, score = str(example[0][0]), str(example[0][1]), float(example[0][2])
                self.model.zero_grad()
                pred, relu = self.model(s1, s2)
                if i % 1000 == 0:
                    pc, items = self.all_similarity()
                    print("iteration {0}, dev pearson coefficient: {1}".format(i, pc))
                loss = loss_func(pred[0], V(score))
                total_loss += loss.item()
                loss.backward()
                opt_func.step()

            print("epoch {0}, running_loss: {1}".format(e, total_loss/self.num_train_examples))

    def train(self, loss_func=nn.MSELoss(), opt_func="adam", batch_size=1, num_epochs=50):
        print("-------------------------  Training STS Predictor -------------------------")
        start_time = time.time()

        self._setup_model(batch_size)
        if opt_func == "adam":
            opt_func = torch.optim.Adam(self.model.parameters(), betas=(0.7, 0.99))
        else:
            print("you've been absolutely trolled")
        
        self._train(loss_func, opt_func, num_epochs)

        elapsed_time = time.time() - start_time
        print("Trained STS Predictor completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))


def train_sts_predictor():
    """
    Trains an STS Predictor according to specs given as argument, before
        saving and returning the trained model.
    """
    sts_data = args.data_dir + '/stsbenchmark/'
    train_di = STSDataIterator(sts_data+'train_data.pkl', args.batch_size)
    dev_di = STSDataIterator(sts_data+'dev_data.pkl', 1)
    test_di = STSDataIterator(sts_data+'test_data.pkl', 1)
    word_embeddings = api.load(args.word_embedding_source)

    predictor = STS('sts_predictor_1', args.saved_models, word_embeddings, args.embedding_dim, train_di, dev_di=dev_di, test_di=test_di)
    predictor.train(batch_size=args.batch_size, num_epochs=args.num_epochs)
    predictor.save()

    print(predictor.model.encoder.dep_freq)
    print(predictor.model.encoder.rare_dependencies)

    return predictor 


def test_sts_predictor():
    sts_data = args.data_dir + '/stsbenchmark/'
    test_di = STSDataIterator(sts_data+'dev_data.pkl', 1, randomise=False)
    word_embeddings = api.load(args.word_embedding_source)
    
    predictor = STS(args.model_name, args.saved_models, word_embeddings, args.embedding_dim, train_di=None, dev_di=None, test_di=test_di)
    pc, info = predictor.all_similarity()
    print(info[:50]) # printing the worst 50 predictions
    print(pc)


if __name__ == "__main__":
    train_sts_predictor()
    # test_sts_predictor()
