import numpy as np
import pickle
import sys

from pathlib import Path


class SentEvalDataReader():
    def __init__(self, data_file):
        self.data_file = data_file

    def read(self):
        self.train_data, self.val_data, self.test_data = [], [], []
        with Path(self.data_file).open('r') as f:
            for i, l in enumerate(f.readlines()):
                l = l.split('\t')
                if l[0] == "tr":
                    self.train_data.append([l[1], l[2]])
                elif l[0] == "va":
                    self.val_data.append([l[1], l[2]])
                elif l[0] == "te":
                    self.test_data.append([l[1], l[2]])
        
        return self.train_data, self.val_data, self.test_data


if __name__ == "__main__":
    data_file = sys.argv[1]
    dr = SentEvalDataReader(data_file)
    train_data, val_data, test_data = dr.read()
    print(len(train_data))
    print(len(val_data))
    print(len(test_data))