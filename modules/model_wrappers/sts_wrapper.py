import torch

from scipy.stats.stats import pearsonr, spearmanr

from modules.models import create_sts_predictor
from modules.utilities import EarlyStopping
from modules.utilities.imports import *

from .base_wrapper import BaseWrapper

__all__ = ["STSWrapper"]


class STSWrapper(BaseWrapper):
    def __init__(self, name, saved_models, embedding_dim, vocab, encoder_model, encoder_args, predictor_model, layers, drops, train_di, val_di, test_di):
        """
        A class for training a Semantic-Textual Similarity predictor.
        """
        self.name = name
        self.saved_models = saved_models
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.encoder_model = encoder_model
        self.encoder_args = encoder_args
        self.predictor_model = predictor_model
        self.train_di, self.val_di, self.test_di = train_di, val_di, test_di
        self.layers, self.drops = layers, drops
        self.create_model()

    def save(self):
        self.save_model()
        self.save_encoder()
        self.save_vocab()

    def save_model(self):
        path = self.saved_models + f'/{self.name}.pt'
        torch.save(self.model.state_dict(), path)

    def save_encoder(self):
        path = self.saved_models + f'/{self.name}_encoder.pt'
        torch.save(self.model.encoder.state_dict(), path)
    
    def save_vocab(self):
        path = self.saved_models + f'/{self.name}_vocab.pkl'
        pickle.dump(self.vocab, Path(path).open('wb'))
    
    def load(self):
        path = self.saved_models + f'/{self.name}.pt'
        self.model.load_state_dict(torch.load(path))

    def load_encoder(self):
        path = self.saved_models + f'/{self.name}_encoder.pt'
        self.model.encoder.load_state_dict(torch.load(path))

    def create_model(self):
        self.model = create_sts_predictor(self.vocab, self.embedding_dim, self.encoder_model, self.predictor_model, self.layers, self.drops, *self.encoder_args)

    def avg_val_loss(self, loss_func):
        self.model.eval(); self.model.training = False
        total_loss = 0.0
        for batch in iter(self.val_di):
            X1, X2, Y = batch.x1, batch.x2, batch.y
            preds = self.model(X1, X2)
            total_loss += loss_func(preds.reshape(-1), Y)
        
        return total_loss / len(self.val_di)

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
        
        return round(pearsonr(preds, scores)[0], 3), round(spearmanr(preds, scores)[0], 3)

    def predict_all(self, to_predict, load=False):
        if load:
            self.create_model()
            self.load()
        self.model.eval(); self.model.training = False
        preds = []
        for eg in to_predict:
            x1 = torch.tensor([self.vocab.stoi[t] for t in eg.x1].reshape(len(eg.x1), 1))
            x2 = torch.tensor([self.vocab.stoi[t] for t in eg.x2].reshape(len(eg.x2), 1))
            pred = torch.softmax(self.model(x1, x2), dim=1)
            preds.append(pred.item())
        
        return preds

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
            avg_val_loss = self.avg_val_loss(loss_func)
            p, s = self.test_correlation()
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
