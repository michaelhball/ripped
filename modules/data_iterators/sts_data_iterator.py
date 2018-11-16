import numpy as np
import pickle

from pathlib import Path

from utilities import randomise


class STSDataIterator(): # this has to be created for each dataset e.g. one each for train, dev, test
    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = pickle.load(Path(data_path).open('rb')) # should be an np array??
        self.num_examples = len(self.data)
        self.reset()
    
    def __len__(self):
        return self.n

    def __iter__(self):
        self.i, self.iter = 0, 0
        while self.i < self.n - 1:
            batch = self.get_batch(self.i)
            self.i += 1
            yield batch

    def batchify(self, data):
        nb = self.num_examples // self.batch_size
        data = np.array(data[:nb*self.batch_size]) # [num_examples,3]
        data = data.reshape(-1, self.batch_size, data.shape[1]) # [num_batches, bs, 3]

        return data

    def get_batch(self, i):
        return self.batched_data[i]
    
    def reset(self):
        self.data = randomise(self.data)
        self.batched_data = self.batchify(self.data)
        self.n = len(self.batched_data)
        self.i, self.iter = 0, 0