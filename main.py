import argparse
import gensim.downloader as api
import pickle
import spacy
import time
import torch
import torch.nn as nn

from pathlib import Path
from scipy.stats.stats import pearsonr

from modules.data_iterators import EasyIterator, STSDataIterator
from modules.models import create_sts_predictor, create_cosine_dist_sts_predictor
from modules.utilities import my_dependencies, plot_train_test_loss, V


parser = argparse.ArgumentParser(description='PyTorch Dependency-Parse Encoding Model')
parser.add_argument('--word_embedding_source', type=str, default='glove-wiki-gigaword-50', help='word embedding source to use')
parser.add_argument('--word_embedding', type=str, default='glove_50', help='name of default word embeddings to use')
parser.add_argument('--saved_models', type=str, default='./models', help='directory to save/load models')
parser.add_argument('--data_dir', type=str, default='./data', help='directory where data is stored')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--num_epochs', type=int, default=8, help='number of epochs to train for')
parser.add_argument('--model_name', type=str, default='sts_predictor', help='name of model to train')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
args = parser.parse_args()


class STS():
    """
    A class for training an STS predictor.
    """
    def __init__(self, name, saved_models, embedding_dim, dependencies, train_di, test_di):
        # NEED TO TAKE IN MORE HERE TO MAKE IT MORE GENERALISABLE
        self.name = name
        self.saved_models = saved_models
        self.embedding_dim = embedding_dim
        self.dependencies = dependencies
        self.batch_size = 1
        self.train_di, self.test_di = train_di, test_di
        self.create_model()

    def save(self):
        self.save_model()
        self._save_encoder()

    def save_model(self):
        path = self.saved_models + '/{0}.pt'.format(self.name)
        torch.save(self.model.state_dict(), path)

    def save_encoder(self):
        path = self.saved_models + '/{0}_encoder.pt'.format(self.name)
        torch.save(self.model.encoder.state_dict(), path)
    
    def load_model(self, model_name):
        path = self.saved_models + '/{0}.pt'.format(model_name)
        self.model.load_state_dict(torch.load(path))

    def load_encoder(self, encoder_name):
        path = self.saved_models + '/{0}_encoder.pt'.format(encoder_name)
        self.model.encoder.load_state_dict(torch.load(path))
    
    def create_model(self):
        self.model = create_cosine_dist_sts_predictor(self.embedding_dim, self.batch_size, self.dependencies)

    def test_pc(self, load=False):
        if load:
            self.create_model()
            self.load_model()
        self.model.eval()
        self.model.training = False
        preds, scores, info = [], [], []
        for i, example in enumerate(iter(self.test_di)):
            s1, s2, score = example
            pred = self.model(s1, s2)
            preds.append(pred.item())
            scores.append(score/5)
            # info.append((s1, s2, score, pred.item()))
        
        pearson_coefficient = pearsonr(preds, scores)
        # info = sorted(info, key=lambda tup: abs(tup[2]-tup[3]), reverse=True)
        
        return pearson_coefficient#, info
    
    def average_test_loss(self, loss_func, load=False):
        if load:
            self.create_model()
            self.load_model()
        self.model.eval()
        self.model.training = False
        total_loss = 0.0
        for i, example in enumerate(iter(self.test_di)):
            s1, s2, score = example
            pred = self.model(s1, s2)
            loss = loss_func(pred[0], (V(score)-1)/5)
            total_loss += loss.item()
        
        return total_loss / self.test_di.num_examples

    def train(self, loss_func, opt_func, num_epochs=50):
        print("-------------------------  Training STS Predictor -------------------------")
        start_time = time.time()

        train_losses = []
        test_losses = [] # change this to use a dev set that's a partition of the training set.
        for e in range(num_epochs):
            self.model.train()
            self.model.training = True
            total_loss = 0.0
            for i, example in enumerate(iter(self.train_di)):
                self.model.zero_grad()
                s1, s2, score = example
                pred = self.model(s1, s2)
                loss = loss_func(pred[0], (V(score)-1)/5)
                total_loss += loss.item()
                loss.backward()
                opt_func.step()

            avg_train_loss = total_loss / self.train_di.num_examples
            avg_test_loss = self.average_test_loss(loss_func)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            print(f"epoch {e+1}, avg train loss: {avg_train_loss}, avg test loss: {avg_test_loss}")

        elapsed_time = time.time() - start_time
        print("Trained STS Predictor completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        return train_losses, test_losses


# def train_sick_predictor(name):
#     sick_data = args.data_dir + '/sick/'
#     # train_di = STSDataIterator(sick_data+'train.pkl', args.batch_size, args.word_embedding_source)
#     # train_di.save_data(sick_data+'train_data_glove_50.pkl')
#     # test_di = STSDataIterator(sick_data+'test.pkl', 1, args.word_embedding_source)
#     # test_di.save_data(args.data_dir+'/test_data_glove_50.pkl')

#     train_di = EasyIterator(sick_data+'train_data_glove_50.pkl')
#     test_di = EasyIterator(sick_data+'test_data_glove_50.pkl')

#     predictor = STS(name, args.saved_models, word_embeddings, args.embedding_dim, train_di, dev_di=test_di)
#     predictor.train(opt_func="adam", batch_size=args.batch_size, num_epochs=args.num_epochs)

#     # pickle.dump(predictor.model.encoder.unseen_words, Path('sick_unseen_words.pkl').open('wb'))

#     # print(predictor.model.encoder.dep_freq)
#     # print(predictor.model.encoder.rare_dependencies)

#     test_sick_predictor(name)

#     return predictor


def train_model(predictor, num_epochs, loss_func, opt_func, visualise=False, save=False):
    train_losses, test_losses = predictor.train(loss_func, opt_func, num_epochs)
    if visualise:
        plot_train_test_loss(train_losses, test_losses)
    if save:
        predictor.save()


if __name__ == "__main__":

    loss_func = nn.MSELoss()
    embedding_dim = int(args.word_embedding.split('_')[1])
    sick_data = args.data_dir + '/sick'

    train_di = EasyIterator(sick_data + f'/train_data_{args.word_embedding}.pkl')
    test_di = EasyIterator(sick_data + f'/test_data_{args.word_embedding}.pkl')
    opt = torch.optim.SGD
    lr = 0.01

    predictor = STS(args.model_name, args.saved_models, embedding_dim, my_dependencies, train_di, test_di)
    opt_func = opt(predictor.model.parameters(), lr=lr)

    if args.task == "train":
        train_model(predictor, args.num_epochs, loss_func, opt_func, visualise=True)
        print(predictor.test_pc())
