from torchtext import data
from torchtext.data.example import Example

from modules.utilities.imports import *


__all__ = ["IntentClassificationDataReader"]


class IntentClassificationDataReader():
    def __init__(self, file_prefix, file_suffix, text, label, val=True):
        self.train_file = f'{file_prefix}train{file_suffix}'
        self.val_file = f'{file_prefix}val{file_suffix}'
        self.test_file = f'{file_prefix}test{file_suffix}'
        self.full_data = {"train": {}, "val": {}, "test": {}}
        self.fields1 = {'x': ('x', text), 'y': ('y', label)}
        self.fields2 = {'x': text, 'y': label}
        self.val = val

    def read_one(self, data_file, dataset_type="train"):
        pkl_data = pickle.load(Path(data_file).open('rb'))
        examples = [Example.fromdict(x, self.fields1) for x in pkl_data]
        dataset = data.Dataset(examples, fields=self.fields2)
        return dataset

    def read(self):
        self.train_ds = self.read_one(self.train_file, "train")
        self.test_ds = self.read_one(self.test_file, "test")
        if self.val:
            self.val_ds = self.read_one(self.val_file, "val")
        else:
            self.val_ds = self.read_one(self.test_file, "test")
        
        return self.train_ds, self.val_ds, self.test_ds
