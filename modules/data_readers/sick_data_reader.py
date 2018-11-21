import numpy as np
import pickle
import sys

from pathlib import Path


class SICKDataReader():
    def __init__(self, train_file, test_file):
        self.train_file, self.test_file = train_file, test_file
        self.full_data = {"train": {}, "dev": {}, "test": {}}

    def read_one(self, data_file, dataset_type="train"):
        with Path(data_file).open('r') as f:
            data = []
            for i, l in enumerate(f.readlines()):
                l = l.split('\t')
                data.append([l[1], l[2], l[3]]) # s1, s2, score
                self.full_data[dataset_type][l[0]] = {'id': l[0], 's1': l[1], 's2': l[2], 'score': l[3], 'entailment': l[4]}

            data = data[1:]
            for d in data:
                                
            return np.array(data)

    def read(self):
        self.train_data = self.read_one(self.train_file, "train")
        self.test_data = self.read_one(self.test_file, "test")
        
        return self.train_data, self.test_data
    
    def dump(self, dump_file):
        pickle.dump(self.train_data, Path(dump_file+'train.pkl').open('wb'))
        pickle.dump(self.test_data, Path(dump_file+'test.pkl').open('wb'))


if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    dump_folder = sys.argv[3]
    dr = SICKDataReader(train_file, test_file)
    train_data, test_data = dr.read()
    dr.dump(dump_folder)