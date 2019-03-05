import time
import torch

from scipy.stats.stats import pearsonr, spearmanr

from modules.models import create_sts_predictor
from modules.utilities import EarlyStopping

from .base_wrapper import BaseWrapper


class STSWrapper(BaseWrapper):
    """
    A class for training an STS predictor.
    """
    def __init__(self, name, saved_models, embedding_dim, vocab, encoder_model, predictor_model, train_di, val_di, test_di, layers, drops, *args):
        self.name = name
        self.saved_models = saved_models
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.encoder_model = encoder_model
        self.predictor_model = predictor_model
        self.train_di, self.val_di, self.test_di = train_di, val_di, test_di
        self.layers, self.drops = layers, drops
        self.args = args
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
        self.model = create_sts_predictor(self.vocab, self.embedding_dim, self.encoder_model, self.predictor_model, self.layers, self.drops, *self.args)

    def avg_loss(self, loss_func, di):
        self.model.eval(); self.model.training = False
        total_loss = 0.0
        for batch in iter(di):
            X1, X2, Y = batch.x1, batch.x2, batch.y
            preds = self.model(X1, X2)
            total_loss += loss_func(preds.reshape(-1), Y)
        
        return total_loss / len(di)

    def test_correlation(self, load=False):
        if load:
            self.create_model()
            self.load()
        self.model.eval(); self.model.training = False
        preds, scores = [], []
        for batch in iter(self.test_di):
            X1, X2, Y = batch.x1, batch.x2, batch.y
            pred = self.model(X1, X2)
            preds += pred.reshape(-1).tolist()
            scores += Y.float().reshape(-1).tolist()
        
        return pearsonr(preds, scores), spearmanr(preds, scores)

    def train(self, loss_func, opt_func, num_epochs=1000, verbose=True):
        if verbose:
            print("-------------------------  Training STS Predictor -------------------------")
        start_time = time.time()
        train_losses, val_losses, correlations = [], [], []
        early_stopping = EarlyStopping(patience=4, verbose=False)

        for e in range(num_epochs):
            self.model.train(); self.model.training = True
            total_loss, num_batches = 0.0, 0
            
            for batch in iter(self.train_di):
                num_batches += 1
                self.model.zero_grad()
                X1, X2, Y = batch.x1, batch.x2, batch.y
                preds = self.model(X1, X2)
                loss = loss_func(preds.reshape(-1), Y)
                total_loss += loss.item()
                loss.backward()
                opt_func.step()
            
            avg_train_loss = total_loss / num_batches
            avg_val_loss = self.avg_loss(loss_func, self.val_di)
            p, s = self.test_correlation()
            p, s = round(p[0],3), round(s[0],3)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            correlations.append((p, s))

            early_stopping(avg_val_loss, self)
            if early_stopping.early_stop:
                if verbose:
                    print(f"epoch {e+1}, avg train loss: {avg_train_loss}, avg val loss: {avg_val_loss}, test pearson: {p}")
                break

            if verbose:
                print(f"epoch {e+1}, avg train loss: {avg_train_loss}, avg val loss: {avg_val_loss}, test pearson: {p}")

        if verbose:
            elapsed_time = time.time() - start_time
            print("Trained STS Predictor completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        return train_losses, val_losses, correlations
