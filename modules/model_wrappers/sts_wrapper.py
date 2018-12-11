import math
import time
import torch

from scipy.stats.stats import pearsonr, spearmanr

from modules.models import create_cosine_sim_sts_predictor, create_sts_predictor
from modules.utilities import V

from .base_wrapper import BaseWrapper


class STSWrapper(BaseWrapper):
    """
    A class for training an STS predictor.
    """
    def __init__(self, name, saved_models, embedding_dim, batch_size, dependencies, train_di, test_di, encoder_model, layers=None, drops=None):
        self.name = name
        self.saved_models = saved_models
        self.embedding_dim = embedding_dim
        self.dependencies = dependencies
        self.batch_size = batch_size
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
        if self.layers and self.drops:
            self.model = create_sts_predictor(self.embedding_dim, self.batch_size, self.dependencies, self.encoder_model, self.layers, self.drops)
        else:
            self.model = create_cosine_sim_sts_predictor(self.embedding_dim, self.batch_size, self.dependencies, self.encoder_model)

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
            info.append((i, score, pred.item()))
        
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
            loss = loss_func(pred[0], (V(score)-1)/4)
            total_loss += loss.item()
        
        return total_loss / self.test_di.num_examples
    
    def find_lr(self, max_lr, ratio, cut_frac, cut, e, i):
        # /50 is for simulating a batch_size of 50
        t_epoch = e * (len(self.train_di) / 50)
        t = t_epoch + (i // 50)
        p = t / cut if t < cut else 1 - ((t - cut) / (cut * (1 / cut_frac - 1)))

        return max_lr * (1 + p * (ratio - 1)) / ratio
    
    def find_lr_sgdr(self, max_lr, min_lr, e, i):
        # /50 is to simulate a batch_size of 50
        num_batches = len(train_di) / 50
        frac = float(i//50) / num_batches # fraction through current epoch

        # cycle every 2 epochs
        Tcur = frac if e%2==0 else frac + 1
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

    def train(self, loss_func, opt_func, lr_scheme=None, num_epochs=50):
        print("-------------------------  Training STS Predictor -------------------------")
        start_time = time.time()

        if lr_scheme:
            MAX_LR = 0.01
            T = num_epochs * (len(self.train_di) / 50)
            cut_frac = 0.1
            cut = max(1.0, float(math.floor(T * cut_frac)))
            ratio = 32

        train_losses, test_losses = [], []
        correlations = []
        for e in range(num_epochs):
            self.model.train()
            self.model.training = True
            total_loss = 0.0
            for i, example in enumerate(iter(self.train_di)):
                
                if lr_scheme: # Slanted Triangular Learning Rates / SGDR (for ENC-3)
                    if i % 50 == 0: # simulated batch_size of 50
                        opt_func.param_groups[0]['lr'] = self.find_lr(MAX_LR, ratio, cut_frac, cut, e, i)
                        # opt_func.param_groups[0]['lr'] = self.find_lr_sgdr(MAX_LR, MAX_LR*0.001, e, i)

                self.model.zero_grad()
                s1, s2, score = example
                pred = self.model(s1, s2)
                loss = loss_func(pred[0], (V(score)-1)/4)
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
