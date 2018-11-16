import argparse
import gensim.downloader as api
import spacy
import time
import torch
import torch.nn as nn

from pathlib import Path

from data_iterators import STSDataIterator
from models import create_sts_predictor
from utilities import my_dependencies, V


parser = argparse.ArgumentParser(description='PyTorch Dependency-Parse Encoding Model')
parser.add_argument('--word_embedding_source', type=str, default='glove-wiki-gigaword-50', help='word embedding source to use')
parser.add_argument('--embedding_dim', type=int, default=50, help='size of word embeddings used')
parser.add_argument('--saved_models', type=str, default='./saved_models', help='directory to save/load models')
parser.add_argument('--data_dir', type=str, default='../data', help='directory where data is stored')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--model_name', type=str, default='sts_predictor', help='name of model to train')
parser.add_argument('--task', type=str, default='train', help='task out of train/test/evaluate')
args = parser.parse_args()


class STS():
    """
    A class for training an STS predictor.
    """
    def __init__(self, name, saved_models, word_embeddings, embedding_dim, train_di, dev_di=None, test_di=None):
        self.name = name
        self.saved_models = saved_models
        self.word_embeddings = word_embeddings
        self.embedding_dim = embedding_dim
        self.train_di, self.dev_di, self.test_di = train_di, dev_di, test_di
        self.nlp = spacy.load('en')
    
    # NOT SURE IF THIS SAVING AND LOADING WILL WORK USING Path() instead of just strings... have to test it out

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
    
    def similarity(self, input1, input2):
        # loads saved version of itself
        None
    
    def _train(self, loss_func, opt_func, num_epochs):
        self.model.train()
        for e in range(num_epochs):
            n = 0
            total_loss = 0.0
            for i, (inputs, targets) in enumerate(iter(self.data_iterator)):
                self.model.zero_grad() # WHAT DOES THIS DO??
                prediction, relu_prediction = self.model(inputs) # inputs are two sentences as plain text
                # need to normalise prediction to be between 1 and 5

                # NOT SURE WHAT THIS DOES?? THIS IS FROM LMCLASSIFIER TO DO WITH CHECKING ACCURACY ETC AS WE GO I GUESS??
                n += result.size()[0] # idk about this?? need to get indices of all these things below right... printing sizes etc.
                for i, r in enumerate(result):
                    if r.max(0)[1] == targets[i]:
                        num_correct += 1
                
                loss = loss_func(prediction, V(targets)) # targets in the simple case will just be one value
                total_loss += loss.item() # not sure if we need this .item() here
                loss.backward()
                opt_func.step()

            print("epoch {0}, running_loss: {1}".format(e, total_loss/len(self.instances)))

    def train(self, loss_func=nn.MSELoss(), opt_func="adam", batch_size=1, num_epochs=50):
        print("-------------------------  Training STS Predictor -------------------------")
        start_time = time.time()

        layers = [200, 50, 1] # how to fine-tune this?
        self.model = create_sts_predictor(self.nlp, self.word_embeddings, self.embedding_dim, batch_size, my_dependencies, layers)
        if opt_func == "adam":
            opt_func = torch.optim.Adam(self.model.parameters(), betas=(0.7, 0.99))
        else:
            print('trolled')
        # self._train(loss_func, opt_func, NUM_EPOCHS)

        elapsed_time = time.time() - start_time
        print("Trained STS Predictor completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))


def train_sts_predictor():
    """
    Trains an STS Predictor according to specs given as argument, before
        saving and returning the trained model.
    """
    sts_data = args.data_dir + '/stsbenchmark/'
    train_di = STSDataIterator(sts_data+'train_data.pkl', args.batch_size)
    word_embeddings = api.load(args.word_embedding_source)

    predictor = STS(args.model_name, args.saved_models, word_embeddings, args.embedding_dim, train_di, dev_di=None, test_di=None)
    predictor.train(batch_size=args.batch_size, num_epochs=args.num_epochs)
    predictor.save()

    return predictor 


if __name__ == "__main__":
    train_sts_predictor()
