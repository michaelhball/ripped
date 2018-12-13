import pickle
import time
import torch

from pathlib import Path
from random import shuffle
from tqdm import tqdm

from modules.models import create_classifier
from modules.utilities import V

from .base_wrapper import BaseWrapper


class ProbingWrapper(BaseWrapper):
    def __init__(self, name, saved_models, probing_task, train_data, test_data, encoder, layers, drops):
        self.name = name
        self.saved_models = saved_models
        self.probing_task = probing_task
        self.train_data = pickle.load(Path(train_data).open('rb'))
        self.test_data = pickle.load(Path(test_data).open('rb'))
        self.encoder = encoder
        self.layers, self.drops = layers, drops
        self.create_model()
    
    def save(self):
        path = self.saved_models + f'/{self.name}_{self.probing_task}.pt'
        torch.save(self.model.state_dict(), path)
    
    def load(self):
        path = self.saved_models + f'/{self.name}_{self.probing_task}.pt'
        self.model.load_state_dict(torch.load(path))

    def create_model(self):
        path = self.saved_models + f'/{self.name}_encoder.pt'
        self.encoder.load_state_dict(torch.load(path))
        self.encoder.eval()
        self.encoder.training = False
        self.model = create_classifier(self.layers, self.drops)
    
    def avg_val_loss(self, loss_func):
        self.model.eval()
        self.model.training = False
        total_loss = 0.0
        for i, (y, x) in enumerate(self.test_data):
            pred = self.model(self.encoder(x))
            total_loss += loss_func(pred, V(y)).item()

        return total_loss / len(self.test_data)

    def test_accuracy(self, load=False):
        if load:
            self.load()
        self.model.eval()
        self.model.training = False
        total_correct = 0.0
        for i, (y, x) in enumerate(self.test_data):
            self.model.zero_grad()
            pred = self.model(self.encoder(x))
            _, c = torch.max(pred[0], 0)
            if c == y:
                total_correct += 1
        
        return total_correct / len(self.test_data)
    
    def train(self, loss_func, opt_func, num_epochs=10):
        print(f"-------------------------  Training Probing Classifier on {self.probing_task} Task -------------------------")
        start_time = time.time()
        train_losses, val_losses  = [], []
        for e in range(num_epochs):
            self.model.train()
            self.model.training = True
            total_loss = 0.0
            shuffle(self.train_data)
            for i, (y,x) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
                self.model.zero_grad()
                pred = self.model(self.encoder(x))
                loss = loss_func(pred, V(y))
                total_loss += loss.item()
                loss.backward()
                opt_func.step()

            avg_train_loss = total_loss / len(self.train_data)
            avg_val_loss = self.avg_val_loss(loss_func)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            print(f'Epoch {e+1}, average train loss {avg_train_loss}, average val loss: {avg_val_loss}, test accuracy: {self.test_accuracy()}')

        elapsed_time = time.time() - start_time
        print(f"Trained Probing Classifier on {self.probing_task} completed in {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

        return train_losses, val_losses
