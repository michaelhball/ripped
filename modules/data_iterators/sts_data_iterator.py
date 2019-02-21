import numpy as np
import pickle

from pathlib import Path
from random import shuffle

from modules.utilities import randomise, T


class STSDataIterator():
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
            batch = (self.batched_s1s[self.i], self.batched_s2s[self.i], self.batched_scores[self.i])
            self.i += 1
            yield batch

    def fetch_data(self):
        self.data = pickle.load(Path(self.data_file).open('rb'))
        self.num_examples = len(self.data)
        self.s1s = np.array([list(x) for x in self.data.T[0]])
        self.s2s = np.array([list(x) for x in self.data.T[1]])
        self.scores = np.array([list(x) for x in self.data.T[2]])

    def batchify(self):
        self.batched_s1s, self.batched_s2s, self.batched_scores = self.s1s, self.s2s, self.scores
        if self.randomise:
            np.random.seed(42)
            perm = np.random.permutation(len(self.data))
            self.batched_s1s = self.s1s[perm]
            self.batched_s2s = self.s2s[perm]
            self.batched_scores = self.scores[perm]
        nb = self.num_examples // self.batch_size
        self.batched_s1s = self.batched_s1s[:nb*self.batch_size]
        self.batched_s2s = self.batched_s2s[:nb*self.batch_size]
        self.batched_scores = self.batched_scores[:nb*self.batch_size]
        self.batched_s1s = T(self.batched_s1s.reshape(-1, self.batch_size, self.batched_s1s.shape[1]))
        self.batched_s2s = T(self.batched_s2s.reshape(-1, self.batch_size, self.batched_s2s.shape[1]))
        self.batched_scores = T(self.batched_scores.reshape(-1, self.batch_size, self.batched_scores.shape[1]))
    
    def reset(self):
        self.i = 0
        self.batchify()
        self.n = len(self.batched_s1s)
