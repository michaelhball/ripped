import time
import torch

from scipy.stats.stats import pearsonr, spearmanr

from modules.models import create_similarity_predictor
from modules.utilities import V

from .base_wrapper import BaseWrapper


class TuningWrapper(BaseWrapper):
    def __init__(self, name, saved_models, train_di, test_di, encoder, layers, drops):
        self.name = name
        self.saved_models = saved_models
        self.train_di, self.test_di = train_di, test_di
        self.encoder = encoder
        self.layers, self.drops = layers, drops
        self.create_model()
    
    def save(self):
        path = self.saved_models + f'/{self.name}_predictor.pt'
        torch.save(self.model.state_dict(), path)
    
    def load(self):
        path = self.saved_models + f'/{self.name}_predictor.pt'
        self.model.load_state_dict(torch.load(path))

    def create_model(self):
        path = self.saved_models + f'/{self.name}_encoder.pt'
        self.encoder.load_state_dict(torch.load(path))
        self.encoder.eval()
        self.encoder.training = False
        self.model = create_similarity_predictor(self.layers, self.drops)
    
    def avg_test_loss(self, loss_func):
        self.model.eval()
        self.model.training = False
        total_loss = 0.0
        for i, (s1, s2, score) in enumerate(iter(self.test_di)):
            x1, x2 = self.encoder(s1), self.encoder(s2)
            pred = self.model(x1, x2)
            loss = loss_func(pred[0], (V(score)-1)/4)
            total_loss += loss.item()
        
        return total_loss / self.test_di.num_examples

    def test_correlation(self, load=False):
        if load:
            self.create_model()
            self.load()
        self.model.eval()
        self.model.training = False
        preds, scores, info = [], [], []
        for i, (s1, s2, score) in enumerate(iter(self.test_di)):
            x1, x2 = self.encoder(s1), self.encoder(s2)
            pred = self.model(x1, x2)
            preds.append(pred.item())
            scores.append((score-1)/4)
            info.append((i, score, pred.item()))
        
        pearson = pearsonr(preds, scores)
        spearman = spearmanr(preds, scores)
        info = sorted(info, key=lambda tup: abs(tup[1]-tup[2]), reverse=True)
        
        return pearson, spearman, info
    
    def train(self, loss_func, opt_func, num_epochs=10):
        print(f"-------------------------  Tuning Similarity Predictor -------------------------")
        start_time = time.time()
        train_losses, test_losses  = [], []
        for e in range(num_epochs):
            self.model.train()
            self.model.training = True
            total_loss = 0.0
            for i, (s1, s2, score) in enumerate(iter(self.train_di)):
                self.model.zero_grad()
                pred = self.model(self.encoder(s1), self.encoder(s2))
                loss = loss_func(pred[0], (V(score)-1)/4)
                total_loss += loss.item()
                loss.backward()
                opt_func.step()

            avg_train_loss = total_loss / self.train_di.num_examples
            avg_test_loss = self.avg_test_loss(loss_func)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            print(f"epoch {e+1}, avg train loss: {avg_train_loss}, avg test loss: {avg_test_loss}")

        elapsed_time = time.time() - start_time
        print("Tuning Similarity Predictor completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        return train_losses, test_losses
