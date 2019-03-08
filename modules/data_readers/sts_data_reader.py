from torchtext import data
from torchtext.data.example import Example

from modules.utilities.imports import *


__all__ = ["STSDataReader"]


class STSDataReader():
    def __init__(self, train_file, val_file, test_file, text, label):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.fields1 = {'x1': ('x1', text), 'x2': ('x2', text), 'y': ('y', label)}
        self.fields2 = {'x1': text, 'x2': text, 'y': label}

    def read_one(self, data_file, dataset_type="train"):
        pkl_data = pickle.load(Path(data_file).open('rb'))
        examples = [Example.fromdict(x, self.fields1) for x in pkl_data]
        dataset = data.Dataset(examples, fields=self.fields2)
        return dataset

    def read(self):
        self.train_ds = self.read_one(self.train_file, "train")
        self.val_ds = self.read_one(self.val_file, "val")
        self.test_ds = self.read_one(self.test_file, "test")
        return self.train_ds, self.val_ds, self.test_ds
