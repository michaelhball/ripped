import numpy as np
import pickle

from pathlib import Path
from random import shuffle

from modules.utilities import T


class EasyIterator():
    def __init__(self, data_file, batch_size=1, randomise=True):
        self.data_file = data_file
        self.batch_size = batch_size
        self.randomise = randomise
        self.fetch_data()
        self.reset()
    
    def __len__(self):
        return self.n

    def __iter__(self):
        self.reset()
        while self.i < self.n - 1:
            batch = self.batched_data[self.i]
            self.i += 1
            yield batch

    def fetch_data(self):
        self.data = pickle.load(Path(self.data_file).open('rb'))
        self.num_examples = len(self.data)
        print(self.data.shape)
        assert(False)

    def batchify(self, data):
        if self.randomise:
            shuffle(data)
        nb = self.num_examples // self.batch_size
        data = np.array(data[:nb*self.batch_size]) # [num_examples,3]
        # data = data.reshape(-1, self.batch_size, data.shape[1]) # [num_batches, bs, 3]
        data = data.reshape(self.batch_size, -1).T
        data = T(data)
        print(data.shape)
        print(type(data))
        assert(False)

        # data = np.array([[self.stoi[o] for o in p] for p in data])
        # data = np.concatenate(randomise(data))
        # nb = data.shape[0] // self.batch_size
        # data = np.array(data[:nb*self.batch_size])
        # data = data.reshape(self.batch_size, -1).T
        
        return data
    
    def reset(self):
        self.i = 0
        self.batched_data = self.batchify(self.data)
        self.n = len(self.batched_data)
