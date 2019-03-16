from modules.utilities.imports import *
from modules.utilities.imports_torch import *

__all__ = ['InferSentIterator']


class InferSentIterator():
    def __init__(self, data, batch_size, randomise=True):
        self.data = data
        self.num_examples = len(self.data)
        self.bs = batch_size
        self.randomise = randomise
        self.reset()
    
    def __len__(self):
        return self.n

    def __iter__(self):
        self.reset()
        while self.i < self.n - 1:
            batch = self.batched_data[self.i]
            self.i += 1
            yield batch

    def batchify(self, data):
        if self.randomise:
            np.random.shuffle(data)
        nb = math.ceil(len(data) / self.bs)
        batches = []
        for i in range(nb):
            x1s, x2s, ys = [], [], []
            batch_data = data[i*64:(i+1)*64]
            for j in range(len(batch_data)):
                x1s.append(batch_data[j]['x1'])
                x2s.append(batch_data[j]['x2'])
                ys.append(batch_data[j]['y'])
            batches.append({'x1': torch.tensor(x1s), 'x2': torch.tensor(x2s), 'y': torch.tensor(ys)})
        
        return batches

    def reset(self):
        self.i = 0
        self.batched_data = self.batchify(self.data)
        self.n = len(self.batched_data)
