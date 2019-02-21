import numpy as np
import pickle

from pathlib import Path
from random import shuffle

from modules.utilities import randomise, T


class MTLDataIterator():
    def __init__(self, sts_di, nli_di):
        self.sts_di = sts_di
        self.nli_di = nli_di
    
    def __len__(self):
        return self.n

    def __iter__(self):
        self.reset()
        self.sts_count, self.nli_count = 0, 0
        while self.i < self.n - 1:
            
            batch = (self.batched_s1s[self.i], self.batched_s2s[self.i], self.batched_scores[self.i])
            self.i += 1
            yield batch
    
    def reset(self):
        self.i = 0
        self.sts_di.reset()
        self.nli_di.reset()
        self.n = self.sts_di.n + self.nli_di.n
        self.frac = self.sts_di.n // self.nli_di.n

