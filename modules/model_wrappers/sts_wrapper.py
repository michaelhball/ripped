import math
import time
import torch

from scipy.stats.stats import pearsonr, spearmanr
from tqdm import tqdm

from modules.models import create_sts_predictor
from modules.utilities import V

from .base_wrapper import BaseWrapper


class STSWrapper(BaseWrapper):
    """
    A class for training an STS predictor.
    """
    def __init__(self, name, saved_models, embedding_dim, train_di, test_di, encoder_model, layers=None, drops=None):
        self.name = name
        self.saved_models = saved_models
        self.embedding_dim = embedding_dim
        self.train_di, self.test_di = train_di, test_di
        self.encoder_model = encoder_model
        self.layers = layers
        self.drops = drops
        self.create_model()

    def save(self):
        self.save_model()
        self.save_encoder()

    def save_model(self):
        path = self.saved_models + f'/{self.name}.pt'
        torch.save(self.model.state_dict(), path)

    def save_encoder(self):
        path = self.saved_models + f'/{self.name}_encoder.pt'
        torch.save(self.model.encoder.state_dict(), path)
    
    def load(self):
        path = self.saved_models + f'/{self.name}.pt'
        self.model.load_state_dict(torch.load(path))

    def load_encoder(self):
        path = self.saved_models + f'/{self.name}_encoder.pt'
        self.model.encoder.load_state_dict(torch.load(path))
    
    def create_model(self):
        self.model = create_sts_predictor(self.embedding_dim, self.encoder_model, self.layers, self.drops)

    def test_correlation(self, load=False):
        if load:
            self.create_model()
            self.load()
        self.model.eval()
        self.model.training = False
        preds, scores, info = [], [], []
        for i, example in enumerate(iter(self.test_di)):
            s1, s2, score = example
            pred = self.model(s1, s2)
            preds.append(pred.item())
            scores.append((score-1)/4)
            info.append((i, (score-1)/4, pred.item()))
        
        pearson = pearsonr(preds, scores)
        spearman = spearmanr(preds, scores)
        info = sorted(info, key=lambda tup: abs(tup[1]-tup[2]), reverse=True)
        
        return pearson, spearman, info
    
    def avg_test_loss(self, loss_func):
        self.model.eval()
        self.model.training = False
        total_loss = 0.0
        for i, example in enumerate(iter(self.test_di)):
            s1, s2, score = example
            pred = self.model(s1, s2)
            total_loss += loss_func(pred[0], V((score)-1)/4)
        
        return total_loss / self.test_di.num_examples

    def train(self, loss_func, opt_func, num_epochs=50):
        print("-------------------------  Training STS Predictor -------------------------")
        start_time = time.time()
        train_losses, test_losses, correlations = [], [], []
        for e in range(num_epochs):
            self.model.train()
            self.model.training = True
            total_loss = 0.0
            for i, example in tqdm(enumerate(iter(self.train_di)), total=len(self.train_di)):
                self.model.zero_grad()
                s1, s2, score = example
                pred = self.model(s1, s2)
                loss = loss_func(pred[0], V((score)-1)/4)
                total_loss += loss.item()
                loss.backward()
                opt_func.step()
            
            avg_train_loss = total_loss / self.train_di.num_examples
            avg_test_loss = self.avg_test_loss(loss_func)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)

            p, s, _ = self.test_correlation()
            correlations.append((e+1, round(p[0],3), round(s[0],3)))
            path = self.saved_models + f'/{self.name}_{e+1}.pt'
            torch.save(self.model.state_dict(), path)
            path = self.saved_models + f'/{self.name}_encoder_{e+1}.pt'
            torch.save(self.model.encoder.state_dict(), path)

            print(f"epoch {e+1}, avg train loss: {avg_train_loss}, avg test loss: {avg_test_loss}, test pearson: {round(p[0], 3)}")

        elapsed_time = time.time() - start_time
        print(correlations)
        print("Trained STS Predictor completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        return train_losses, test_losses
