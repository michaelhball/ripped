from torchtext import data
from torchtext.data.example import Example

from modules.utilities.imports import *


__all__ == ["IntentClassificationDataReader"]


class IntentClassificationDataReader():
    def __init__(self, file_prefix, file_suffix, text, label):
        self.train_file = f'{file_prefix}train{file_suffix}'
        self.val_file = f'{file_prefix}val{file_suffix}'
        self.test_file = f'{file_prefix}test{file_suffix}'
        self.full_data = {"train": {}, "val": {}, "test": {}}
        self.fields1 = {'x': ('x', text), 'y': ('y', label)}
        self.fields2 = {'x': text, 'y': label}

    def read_one(self, data_file, dataset_type="train"):
        pkl_data = pickle.load(Path(data_file).open('rb'))
        examples = [Example.fromdict(x, self.fields1) for x in pkl_data]
        dataset = data.Dataset(examples, fields=self.fields2)
        self.full_data[dataset_type] = dataset
        return dataset

    def read(self):
        self.train_ds = self.read_one(self.train_file, "train")
        self.val_ds = self.read_one(self.val_file, "val")
        self.test_ds = self.read_one(self.test_file, "test")
        
        return self.train_ds, self.val_ds, self.test_ds


if __name__ == "__main__":
    
    pass

    # WHAT I DID TO FORMAT THE DATA FROM ORIGINAL SOURCE

    # # collate snips data together into single dataset
    # all_data_train, all_data_test = [], []
    # for idx, i in enumerate(intents):
    #     train_data = pickle.load(Path(f'/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/{i}_train.pkl').open('rb'))
    #     test_data = pickle.load(Path(f'/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/{i}_dev.pkl').open('rb'))
    #     all_data_train += [[x, idx] for x in train_data]
    #     all_data_test += [[x, idx] for x in test_data]
    # pickle.dump(all_data_train, Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/all_data_train.pkl').open('wb'))
    # pickle.dump(all_data_test, Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/all_data_test.pkl').open('wb'))


    # # to create balanced train,test,dev (from snips train data only)
    # train_data = pickle.load(Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/all_data_train.pkl').open('rb')) # 13784
    # test_data = pickle.load(Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/all_data_test.pkl').open('rb')) # 700
    # freqs = {i: [] for i in range(len(intents))}
    # for x in train_data:
    #     freqs[x[1]].append(x[0])
    # train_data, val_data, test_data = [], [], []
    # for i in range(len(intents)):
    #     l = freqs[i]
    #     np.random.shuffle(l)
    #     n = len(l)
    #     train, val, test = l[:int(n*0.8)], l[int(n*0.8):int(n*0.9)], l[int(n*0.9):]
    #     train_data += [[x, i] for x in train]
    #     val_data += [[x, i] for x in val]
    #     test_data += [[x, i] for x in test]
    # pickle.dump(train_data, Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/train.pkl').open('wb'))
    # pickle.dump(val_data, Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/val.pkl').open('wb'))
    # pickle.dump(test_data, Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/test.pkl').open('wb'))


    # # create tokenised datasets
    # nlp = spacy.load('en')
    # train = pickle.load(Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/train.pkl').open('rb'))
    # val = pickle.load(Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/val.pkl').open('rb'))
    # test = pickle.load(Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/test.pkl').open('rb'))
    # val = [{'x': [t.text.lower() for t in nlp(x[0])], 'y': x[1]} for x in tqdm(val, total=len(val))]
    # test = [{'x': [t.text.lower() for t in nlp(x[0])], 'y': x[1]} for x in tqdm(test, total=len(test))]
    # train = [{'x': [t.text.lower() for t in nlp(x[0])], 'y': x[1]} for x in tqdm(train, total=len(train))]
    # pickle.dump(train, Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/train_tknsd.pkl').open('wb'))
    # pickle.dump(val, Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/val_tknsd.pkl').open('wb'))
    # pickle.dump(test, Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/test_tknsd.pkl').open('wb'))


    # # # create torchtext 'examples' datasets
    # train = pickle.load(Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/train_tknsd.pkl').open('rb'))
    # val = pickle.load(Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/val_tknsd.pkl').open('rb'))
    # test = pickle.load(Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/test_tknsd.pkl').open('rb'))
    # print(train[0])
    # from torchtext.data.example import Example
    # TEXT = data.Field(sequential=True)
    # LABEL = data.LabelField(dtype=torch.int, use_vocab=False)
    # fields = {'x': TEXT, 'y': LABEL}
    # fields = {'x': ('x', TEXT), 'y': ('y', LABEL)}
    # train_egs = [Example.fromdict(x, fields) for x in train]
    # val_egs = [Example.fromdict(x, fields) for x in val]
    # test_egs = [Example.fromdict(x, fields) for x in test]
    # print(train_egs[0].x, train_egs[0].y)
    # pickle.dump(train_egs, Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/train_egs.pkl').open('wb'))
    # pickle.dump(val_egs, Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/val_egs.pkl').open('wb'))
    # pickle.dump(test_egs, Path('/Users/michaelball/Desktop/Thesis/repo/data/snipsnlu/test_egs.pkl').open('wb'))