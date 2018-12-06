import gensim.downloader as api
import spacy

from random import shuffle

from modules.utilities import EmbeddingNode


class SentEvalDataIterator():
    def __init__(self, data_reader, we_source, type_='tr', randomise=False):
        tr, va, te = data_reader.read()
        self.all_data = {"tr": tr, "va": va, "te": te}
        self.we = api.load(we_source)
        self.randomise = randomise
        self.tokeniser = spacy.load('en')
        self.type = type_
        self.data = self.all_data[self.type]
        self.num_examples = len(self.data)
    
    def __len__(self):
        return self.num_examples

    def __iter__(self):
        self.reset()
        while self.i < self.num_examples - 1:
            example = self.data[self.i]
            self.i += 1
            yield example[0], self.tokenise_sent(example[1])

    def tokenise_sent(self, sentence):
        return EmbeddingNode(self.we, list(self.tokeniser(sentence).sents)[0].root)

    def change_type(self, type_="tr", randomise=False):
        self.type = type_
        self.data = self.all_data[self.type]
        self.num_examples = len(self.data)
        self.randomise = randomise

    def reset(self):
        self.i = 0
        if self.randomise:
            shuffle(self.data)
