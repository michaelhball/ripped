import argparse
import gensim.downloader as api
import math
import pickle
import spacy
import time
import torch
import torch.nn as nn

from pathlib import Path
from scipy.stats.stats import pearsonr

from modules.data_iterators import EasyIterator, STSDataIterator
from modules.models import create_cosine_sim_sts_predictor, create_euclid_sim_sts_predictor
from modules.utilities import all_dependencies, my_dependencies, plot_train_test_loss, V


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
        self.name = name
        self.saved_models = saved_models
        self.embedding_dim = embedding_dim
        self.dependencies = dependencies
        self.batch_size = 1
        self.train_di, self.test_di = train_di, test_di
        self.create_model()

    def save(self):
        self.save_model()
        self.save_encoder()

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
        self.model = create_cosine_sim_sts_predictor(self.embedding_dim, self.batch_size, self.dependencies, encoder_model=3)

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
    
    def find_lr(self, max_lr, ratio, cut_frac, cut, e, i):
        t_epoch = e * (len(self.train_di) / 100) # simulating batch_size of 20
        t = t_epoch + (i // 100) # simulating batch_size of 20
        p = t / cut if t < cut else 1 - ((t - cut) / (cut * (1 / cut_frac - 1)))

        return max_lr * (1 + p * (ratio - 1)) / ratio
    
    def find_lr_sgdr(self, max_lr, min_lr, e, i):
        num_batches = len(train_di) / 50 # simulating batch_size of 50
        frac = float(i//50) / num_batches # fraction through current epoch

        # cycle every 2 epochs
        if e % 2 == 0:
            Tcur = frac
        else:
            Tcur = frac + 1
        Ti = 2
        return min_lr + 0.5*(max_lr-min_lr)*(1+math.cos((Tcur/Ti)*math.pi))

        # 3 different length cycles.
        if e == 0:
            Tcur = frac
            Ti = 1
        elif e < 4:
            Tcur = frac + e - 1
            Ti = 4
        else:
            Tcur = frac + e - 4
            Ti = 10

        return min_lr + 0.5*(max_lr-min_lr)*(1+math.cos((Tcur/Ti)*math.pi))

    def train(self, loss_func, opt_func, num_epochs=50):
        print("-------------------------  Training STS Predictor -------------------------")
        start_time = time.time()

        # MAX_LR = 0.1
        # T = num_epochs * (len(self.train_di) / 100) # simulating batch_size of 20
        # cut_frac = 0.1
        # cut = max(1.0, float(math.floor(T * cut_frac)))
        # ratio = 32

        train_losses = []
        test_losses = [] # change this to use a dev set that's a partition of the training set.
        for e in range(num_epochs):
            self.model.train()
            self.model.training = True
            total_loss = 0.0
            for i, example in enumerate(iter(self.train_di)):
                
                # Slanted Triangular Learning Rates / SGDR
                # if i % 50 == 0: # i.e. simulated batch size of 50
                    # opt_func.param_groups[0]['lr'] = self.find_lr(MAX_LR, ratio, cut_frac, cut, e, i)
                    # opt_func.param_groups[0]['lr'] = self.find_lr_sgdr(MAX_LR, MAX_LR*0.001, e, i)

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


def train_model(predictor, num_epochs, loss_func, opt_func, visualise=False, save=False):
    train_losses, test_losses = predictor.train(loss_func, opt_func, num_epochs)
    if visualise:
        plot_train_test_loss(train_losses, test_losses, save_file=f'./data/sick/loss_plots/{args.model_name}.png')
    if save:
        predictor.save()


if __name__ == "__main__":

    # loss_func = nn.MSELoss()
    # embedding_dim = int(args.word_embedding.split('_')[1])
    # sick_data = args.data_dir + '/sick'

    # train_di = EasyIterator(sick_data + f'/train_data_{args.word_embedding}.pkl')
    # test_di = EasyIterator(sick_data + f'/test_data_{args.word_embedding}.pkl')
    # opt = torch.optim.Adam
    # lr = 0.001
    # weight_decay = 0
    # params = {} # params to modify encoding model

    # predictor = STS(args.model_name, args.saved_models, embedding_dim, all_dependencies, train_di, test_di)
    # opt_func = opt(predictor.model.parameters(), lr=lr, weight_decay=weight_decay)

    # # pg1 = [{'params': predictor.model.encoder.lstm.parameters(), 'lr': lr}]
    # # pg2 = [{'params': ps, 'lr': lr} for ps in predictor.model.encoder.params.parameters()]
    # # param_groups = pg1 + pg2
    # # opt_func = opt(param_groups)

    # if args.task == "train":
    #     train_model(predictor, args.num_epochs, loss_func, opt_func, visualise=True, save=True)
    #     print(predictor.test_pc())


    # from modules.utilities import tokenise
    # from collections import defaultdict

    # vocab = pickle.load(Path('./data/sick/vocab.pkl').open('rb'))
    # string2idx = defaultdict(lambda: 0, {v: k for k,v in enumerate(vocab)})

    # train_data = pickle.load(Path('./data/sick/train_data_indexed.pkl').open('rb'))
    # test_data = pickle.load(Path('./data/sick/test_data_indexed.pkl').open('rb'))
    # these are the datasets we can use as input to the model. Need to add embedding to the model yet though.
    
    train_data = pickle.load(Path('./data/sick/train_data_indexed.pkl').open('rb'))
    test_data = pickle.load(Path('./data/sick/test_data_indexed.pkl').open('rb'))

    total_len = 0
    for x in train_data:
        total_len += len(x[0]) + len(x[1])
    print(total_len / (len(train_data) * 2))
    total_len = 0
    for y in test_data:
        total_len += len(y[0]) + len(y[1])
    print(total_len / (len(test_data) * 2))
