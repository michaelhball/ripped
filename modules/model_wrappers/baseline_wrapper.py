import pickle
import time
import torch

from pathlib import Path
from random import shuffle
from scipy.stats.stats import pearsonr, spearmanr

from modules.baseline_models import create_pooling_baseline
from modules.utilities import V

from .base_wrapper import BaseWrapper


class BaselineWrapper(BaseWrapper):
    def __init__(self, name, saved_models, train_data, test_data, layers, drops, pool_type):
        self.name = name
        self.saved_models = saved_models
        self.train_data = pickle.load(Path(train_data).open('rb'))
        self.test_data = pickle.load(Path(test_data).open('rb'))
        self.layers = layers
        self.drops = drops
        self.pool_type = pool_type
        self.create_model()
    
    def save(self):
        path = self.saved_models + f'/baseline_{self.name}.pt'
        torch.save(self.model.state_dict(), path)
    
    def load(self):
        path = self.saved_models + f'/baseline_{self.name}.pt'
        self.model.load_state_dict(torch.load(self.model.state_dict(), path))
    
    def create_model(self):
        self.model = create_pooling_baseline(self.layers, self.drops, self.pool_type)

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
            scores.append(score-1/4)
            info.append((i, score, pred.item()))
        
        pearson = pearsonr(preds, scores)
        spearman = spearmanr(preds, scores)
        info = sorted(info, key=lambda tup: abs(tup[1]-tup[2]), reverse=True)
        
        return pearson, spearman, info
    
    def train(self, loss_func, opt_func, num_epochs=10):
        print("-------------------------  Training Baseline Predictor -------------------------")
        start_time = time.time()
        train_losses = []
        for e in range(num_epochs):
            self.model.train()
            self.model.training = True
            total_loss = 0.0
            shuffle(self.train_data)
            for i, example in enumerate(self.test_data):
                self.model.zero_grad()
                s1, s2, score = example
                pred = self.model(s1, s2)
                loss = loss_func(pred[0], V((score-1)/4))
                total_loss += loss.item()
                loss.backward()
                opt_func.step()

            avg_train_loss = total_loss / len(self.train_data)
            train_losses.append(avg_train_loss)
            print(f'epoch {e+1}, average train loss: {avg_train_loss}')

        elapsed_time = time.time() - start_time
        print("Trained Baseline Predictor completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        return train_losses
