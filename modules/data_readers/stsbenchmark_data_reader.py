import csv
import numpy as np
import pickle

from pathlib import Path

from modules.utilities import randomise


class STSBenchmarkDataReader():
    def __init__(self, train_csv, dev_csv, test_csv):
        self.train_csv, self.dev_csv, self.test_csv = train_csv, dev_csv, test_csv
        self.full_data = {"train": {}, "dev": {}, "test": {}}

    def read_one(self, csv_file, dataset_type="train"):
        with open(csv_file) as f:
            csv_reader = csv.reader(f, delimiter='\t')
            data = []
            for i, row in enumerate(csv_reader):
                r = row[:7] # ignoring extra information
                if len(r) == 6:
                    r = r[:5] + r[5].split("\t")
                self.full_data[dataset_type][i] = {'genre': r[0], 'filename': r[1], 'year': r[2], 'score': r[3], 's1': r[4], 's2': r[5]}
                data.append([r[5],r[6], r[4]])
            
            return np.array(data)

    def read(self):
        self.train_data = self.read_one(self.train_csv, "train")
        self.dev_data = self.read_one(self.dev_csv, "dev")
        self.test_data = self.read_one(self.test_csv, "test")
        
        return self.train_data, self.dev_data, self.test_data
    
    def dump(self):
        pickle.dump(self.full_data, Path('../../data/stsbenchmark/full_data.pkl').open('wb'))
        pickle.dump(train_data, Path('../../data/stsbenchmark/train_data.pkl').open('wb'))
        pickle.dump(dev_data, Path('../../data/stsbenchmark/dev_data.pkl').open('wb'))
        pickle.dump(test_data, Path('../../data/stsbenchmark/test_data.pkl').open('wb'))