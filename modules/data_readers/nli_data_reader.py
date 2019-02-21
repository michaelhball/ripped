import json
import sys

from pathlib import Path


class NLIDataReader():
    def __init__(self, train_file, val_file, test_file):
        self.train_file, self.val_file, self.test_file = train_file, val_file, test_file
        self.full_data = {"train": {}, "val": {}, "test": {}}

    def read_one(self, data_file, dataset_type="train"):
        with Path(data_file).open('r') as f:
            columns = f.readline()
            data = []
            for i, l in enumerate(f.readlines()):
                l = l.split('\t')
                data.append([l[5], l[6], l[0]])
                self.full_data[dataset_type][l[8]] = l

            return data

    def read(self):
        self.train_data = self.read_one(self.train_file, "train")
        self.val_data = self.read_one(self.val_file, "val")
        self.test_data = self.read_one(self.test_file, "test")
        
        return self.train_data, self.val_data, self.test_data
    
    def dump(self, dump_folder):
        json.dump(self.train_data, Path(dump_folder+'train.json').open('w'))
        json.dump(self.val_data, Path(dump_folder+'val.json').open('w'))
        json.dump(self.test_data, Path(dump_folder+'test.json').open('w'))


if __name__ == "__main__":
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]
    dump_folder = sys.argv[4]
    # dr = NLIDataReader(train_file, val_file, test_file)
    # train_data, val_data, test_data = dr.read()
    # dr.dump(dump_folder)
    # test_data = json.load(Path('./data/multinli/objects/test.json').open('r'))