import numpy as np
import pickle

from pathlib import Path
from random import shuffle

from modules.utilities import randomise, tokenise


# THIS CAN BE CLEANED UP A BIT TO TAKE IN A DATA_READER... IT'S JUST THE EASY_ITERATOR THAT ONLY TAKES IN A FILE
# , AND THIS ONE TAKES IN THE FILE CONTAINING LIST OF EMBEDDING_NODE PAIRS.
class STSDataIterator(): # this has to be created for each dataset e.g. one each for train, dev, test
    def __init__(self, data_path, batch_size, we_source, randomise=True):
        self.randomise = randomise
        self.batch_size = batch_size
        self.we_source = we_source
        self.data_path = data_path
        self.data = pickle.load(Path(data_path).open('rb'))
        self.num_examples = len(self.data)
        self.reset()
    
    def __len__(self):
        return self.n

    def __iter__(self):
        self.reset()
        while self.i < self.n - 1:
            batch = self.get_batch(self.i)
            self.i += 1
            yield batch
    
    def save_data(self, save_file):
        pickle.dump(self.data, Path(save_file).open('wb'))

    def batchify(self, data):
        if self.randomise:
            data = shuffle(data)
        nb = self.num_examples // self.batch_size
        data = np.array(data[:nb*self.batch_size]) # [num_examples,3]
        data = data.reshape(-1, self.batch_size, data.shape[1]) # [num_batches, bs, 3]

        return data

    def get_batch(self, i):
        return self.batched_data[i]
    
    def reset(self):
        self.data = tokenise(self.we_source, self.data)
        # self.batched_data = self.batchify(self.data)
        # self.n = len(self.batched_data)
        # self.i, self.iter = 0, 0