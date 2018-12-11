import pickle
import time
import torch

from pathlib import Path
from random import shuffle

from modules.models import create_classifier
from modules.utilities import V

from .base_wrapper import BaseWrapper


class DownstreamWrapper(BaseWrapper):
    def __init__(self, name, saved_models, task, train_data, test_data, encoder, layers, drops):
        self.name = name
        self.saved_models = saved_models
        self.task = task
        self.train_data = pickle.load(Path(train_data).open('rb'))
        self.test_data = pickle.load(Path(test_data).open('rb'))
        self.encoder = encoder
        self.layers, self.drops = layers, drops
        self.create_model()
    
    def save(self):
        path = self.saved_models + f'/{self.name}_{self.task}.pt'
        torch.save(self.model.state_dict(), path)
    
    def load(self):
        path = self.saved_models + f'/{self.name}_{self.task}.pt'
        self.model.load_state_dict(torch.load(path))

    def create_model(self):
        path = self.saved_models + f'/{self.name}_encoder.pt'
        self.encoder.load_state_dict(torch.load(path))
        self.encoder.eval()
        self.encoder.training = False
        self.model = create_classifier(self.layers, self.drops)
    
    def avg_test_loss(self, loss_func):
        self.model.eval()
        self.model.training = False
        total_loss = 0.0
        for i, (x, y) in enumerate(self.test_data):
            pred = self.model(self.encoder(x))
            total_loss += loss_func(pred, V(y)).item()

        return total_loss / len(self.test_data)

    def test_accuracy(self, load=False):
        if load:
            self.load()
        self.model.eval()
        self.model.training = False
        total_correct = 0.0
        for i, (x, y) in enumerate(self.test_data):
            self.model.zero_grad()
            pred = self.model(self.encoder(x))
            _, c = torch.max(pred[0], 0)
            if c == y:
                total_correct += 1
        
        return total_correct / len(self.test_data)
    
    def train(self, loss_func, opt_func, num_epochs=10):
        print(f"-------------------------  Training Downstream on {self.task} -------------------------")
        start_time = time.time()
        train_losses, test_losses  = [], []
        for e in range(num_epochs):
            self.model.train()
            self.model.training = True
            total_loss = 0.0
            shuffle(self.train_data)
            for i, (x, y) in enumerate(self.train_data):
                self.model.zero_grad()
                pred = self.model(self.encoder(x))
                loss = loss_func(pred, V(y))
                total_loss += loss.item()
                loss.backward()
                opt_func.step()

            avg_train_loss = total_loss / len(self.train_data)
            avg_test_loss = self.avg_test_loss(loss_func)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            print(f'Epoch {e+1}, average train loss {avg_train_loss}, average test loss: {avg_test_loss}, test accuracy: {self.test_accuracy()}')

        elapsed_time = time.time() - start_time
        print(f"Trained Downstream on {self.task} completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        return train_losses, test_losses
