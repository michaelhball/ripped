import pickle
import time
import torch

from pathlib import Path
from random import shuffle
from scipy.stats.stats import pearsonr, spearmanr
from tqdm import tqdm

from modules.baseline_models import create_lstm_baseline, create_lstm_avg_baseline, create_lstm_max_baseline, create_pooling_baseline
from modules.utilities import V

from .base_wrapper import BaseWrapper


class BaselineWrapper(BaseWrapper):
    def __init__(self, name, saved_models, train_data, test_data, layers, drops, baseline_type, embedding_dim, num_layers=1):
        self.name = name
        self.saved_models = saved_models
        self.train_data = pickle.load(Path(train_data).open('rb'))
        self.test_data = pickle.load(Path(test_data).open('rb'))
        self.layers, self.drops = layers, drops
        self.baseline_type = baseline_type
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.create_model()
    
    def save(self):
        path = self.saved_models + f'/baseline_{self.name}.pt'
        torch.save(self.model.state_dict(), path)
        if self.baseline_type == "lstm":
            path = self.saved_models + f'/baseline_{self.name}_encoder.pt'
            torch.save(self.model.encoder.state_dict(), path)
    
    def load(self):
        path = self.saved_models + f'/baseline_{self.name}.pt'
        self.model.load_state_dict(torch.load(self.model.state_dict(), path))
    
    def create_model(self):
        if self.baseline_type == "lstm_avg":
            self.model = create_lstm_avg_baseline(self.embedding_dim, self.num_layers, self.layers, self.drops)
        elif self.baseline_type == "lstm_max":
            self.model = create_lstm_max_baseline(self.embedding_dim, self.num_layers, self.layers, self.drops)
        elif self.baseline_type == "lstm":
            self.model = create_lstm_baseline(self.embedding_dim, self.num_layers, self.layers, self.drops)
        else:
            self.model = create_pooling_baseline(self.layers, self.drops, self.baseline_type.split('_')[1])

    def test_correlation(self, load=False):
        if load:
            self.create_model()
            self.load_model()
        self.model.eval()
        self.model.training = False
        preds, scores, info = [], [], []
        for i, example in enumerate(self.test_data):
            s1, s2, score = example
            pred = self.model(s1, s2)
            preds.append(pred.item())
            scores.append((score-1)/4)
            info.append((i, score, pred.item()))
        
        pearson = pearsonr(preds, scores)
        spearman = spearmanr(preds, scores)
        info = sorted(info, key=lambda tup: abs(tup[1]-tup[2]), reverse=True)
        
        return pearson, spearman, info
    
    def avg_test_loss(self, loss_func, load=False):
        if load:
            self.create_model()
            self.load_model()
        self.model.eval()
        self.model.training = False
        total_loss = 0.0
        for i, example in enumerate(self.test_data):
            s1, s2, score = example
            pred = self.model(s1, s2)
            total_loss += loss_func(pred[0], V((score-1)/4))

        return total_loss / len(self.test_data)
    
    def batcher(self, data, bs):
        import numpy as np
        n = len(data) // bs
        batched_data = np.array([data[i:i+bs] for i in range(0, len(data), bs)]).reshape((n,3,-1)) # n x bs x 3
        # for x in batched_data:
        #     yield x
    
    def train(self, loss_func, opt_func, num_epochs=10):
        print("-------------------------  Training Baseline Predictor -------------------------")
        start_time = time.time()
        train_losses, test_losses = [], []
        BS = 25
        for e in range(num_epochs):
            self.model.train()
            self.model.training = True
            total_loss = 0.0
            shuffle(self.train_data)
            for i, example in tqdm(enumerate(self.train_data), total=len(self.train_data)):
                self.model.zero_grad()
                s1, s2, score = example
                pred = self.model(s1, s2)
                loss = loss_func(pred[0], V((score-1)/4))
                total_loss += loss.item()
                loss.backward()
                opt_func.step()

            avg_train_loss = total_loss / len(self.train_data)
            train_losses.append(avg_train_loss)
            test_loss = self.avg_test_loss(loss_func)
            test_losses.append(test_loss)
            print(f'epoch {e+1}, average train loss: {avg_train_loss}, average test loss: {test_loss}')
            # p,s,i = self.test_correlation()
            # print(p)

        elapsed_time = time.time() - start_time
        print("Trained Baseline Predictor completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        return train_losses, test_losses
