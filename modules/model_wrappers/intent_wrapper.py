import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

from tqdm import tqdm

from modules.models import create_intent_classifier
from modules.utilities import EarlyStopping

from .base_wrapper import BaseWrapper


class IntentWrapper(BaseWrapper):
    """
    A class for training intent classification model.
    """
    def __init__(self, name, saved_models, embedding_dim, vocab, encoder_model, train_di, val_di, test_di, layers, drops=None):
        self.name = name
        self.saved_models = saved_models
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.encoder_model = encoder_model
        self.train_di, self.val_di, self.test_di = train_di, val_di, test_di
        self.layers, self.drops = layers, drops
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
        path = self.saved_models + f'/{self.name}_best_checkpoint.pt'
        self.model.load_state_dict(torch.load(path))

    def load_encoder(self):
        path = self.saved_models + f'/{self.name}_encoder.pt'
        self.model.encoder.load_state_dict(torch.load(path))
    
    def create_model(self):
        if self.encoder_model == "lstm":
            args = [self.embedding_dim, self.embedding_dim, 1]
        elif self.encoder_model == "we_pool":
            args = [self.embedding_dim]
        else:
            print('there are no classes set up for this encoder type yet')
        self.model = create_intent_classifier(self.encoder_model, self.vocab, self.layers, self.drops, *args)

    def avg_val_loss(self, loss_func):
        self.model.eval(); self.model.training = False
        total_loss = 0.0
        for example in self.val_di:
            X, Y = example.x, example.y
            total_loss += loss_func(self.model(X), Y.long())
        
        return total_loss / len(self.val_di)
    
    def test_accuracy(self, load=False):
        if load:
            self.create_model()
            self.load()
        self.model.eval(); self.model.training = False
        num_correct, num_examples = 0, 0
        for batch in self.test_di:
            X, Y = batch.x, batch.y
            pred = self.model(X)
            pred_idx = torch.max(pred, dim=1)[1]
            num_correct += np.sum(pred_idx.data.numpy() == Y.data.numpy())
            num_examples += len(Y)
        
        return float(num_correct/num_examples)
    
    def lr_finder(self, loss_func, opt_func, min_lr=1e-7, max_lr=0.2, n=85):
        """
        'LR range test' from Leslie Smith: https://arxiv.org/pdf/1506.01186.pdf
        """
        q = math.pow((max_lr/min_lr), float(1/n))
        iterator = iter(self.train_di)
        lrs, losses = [], []
        for i in range(n):
            
            lr = min_lr * math.pow(q, i)
            lrs.append(lr)
            for pg in opt_func.param_groups:
                pg['lr'] = lr

            opt_func.zero_grad()
            example = next(iterator)
            pred = self.model(example.x)
            loss = loss_func(pred, example.y.long())
            losses.append(loss.item())
            loss.backward()
            opt_func.step()

        plt.plot(lrs, losses)
        plt.xlabel('lr'); plt.ylabel('loss')
        plt.xticks(np.arange(0, max_lr, 0.01))
        plt.show()

    def repeat_trainer(self, loss_func, opt, lr, betas, weight_decay, num_epochs=100, k=10):
        model_name = self.name
        for i in range(k):
            self.name = model_name + f'_{i}'
            self.create_model()
            opt_func = opt(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
            self.train(loss_func, opt_func, num_epochs=num_epochs, verbose=False)

        accuracies = []
        for i in range(k):
            self.name = model_name + f'_{i}'
            accuracies.append(self.test_accuracy(load=True))

        return np.mean(accuracies), np.std(accuracies)

    def train(self, loss_func, opt_func, num_epochs=1000, verbose=True):
        if verbose:
            print("-------------------------  Training Intent Classifier -------------------------")
        start_time = time.time()
        train_losses, val_losses = [], []
        early_stopping = EarlyStopping(patience=4, verbose=False) # EXPERIMENT

        for e in range(num_epochs):
            self.model.train(); self.model.training = True
            total_loss = 0.0
            
            it = tqdm(self.train_di, total=len(self.train_di)) if verbose else self.train_di
            for example in it:
                X, Y = example.x, example.y
                self.model.zero_grad()
                preds = self.model(X)
                loss = loss_func(preds, Y.long())
                total_loss += loss.item()
                loss.backward()
                opt_func.step()
            
            avg_train_loss = total_loss / len(self.train_di)
            avg_val_loss = self.avg_val_loss(loss_func)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            early_stopping(avg_val_loss, self.model, self.saved_models, self.name)
            if early_stopping.early_stop:
                if verbose:
                    print(f"epoch {e+1}, avg train loss: {avg_train_loss}, avg val loss: {avg_val_loss}, test accuracy: {self.test_accuracy()}")
                break   
            
            if verbose:
                print(f"epoch {e+1}, avg train loss: {avg_train_loss}, avg val loss: {avg_val_loss}, test accuracy: {self.test_accuracy()}")
        
        # print(f'accuracy: {self.test_accuracy()}')

        if verbose:
            elapsed_time = time.time() - start_time
            print("Training intent classifier completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        return train_losses, val_losses
