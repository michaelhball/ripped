import pickle

from pathlib import Path
from random import shuffle


class EasyIterator():
    def __init__(self, data_file, batch_size=1, randomise=True):
        self.data_file = data_file
        self.batch_size = batch_size
        self.randomise = randomise
        self.fetch_data()
        self.reset()
    
    def __len__(self):
        return self.num_examples

    def __iter__(self):
        self.reset()
        while self.i < self.num_examples - 1:
            example = self.data[self.i]
            self.i += 1
            yield example

    def fetch_data(self):
        self.data = pickle.load(Path(self.data_file).open('rb'))
        self.num_examples = len(self.data)
    
    def reset(self):
        self.i = 0
        if self.randomise:
            shuffle(self.data)
