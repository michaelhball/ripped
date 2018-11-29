import gensim.downloader as api
import numpy as np
import spacy

from collections import Counter, defaultdict
from pathlib import Path


class EmbeddingNode():
    def __init__(self, we, node):
        self.text = node.text.lower()
        self.dep = node.dep_
        self.representation = None
        self.embedding = we[self.text] if self.text in we else we['unk']
        self.children = [EmbeddingNode(we, c) for c in node.children]


def tokenise_and_embed(we_source, data):
    we = api.load(we_source)
    nlp = spacy.load('en')
    tokenised = []
    if type(data) in (list, np.ndarray):
        for d in data:
            s1 = EmbeddingNode(we, list(nlp(str(d[0])).sents)[0].root)
            s2 = EmbeddingNode(we, list(nlp(str(d[1])).sents)[0].root)
            score = float(d[2])
            tokenised.append([s1, s2, score])
    
    return tokenised


def tokenise(data):
    nlp = spacy.load('en')
    tokenised = []
    for x in data:
        s1 = [t.text.lower() for t in nlp(str(x[0]))]
        s2 = [t.text.lower() for t in nlp(str(x[1]))]
        score = float(x[2])
        tokenised.append([s1, s2, score])
    
    return tokenised


def create_vocab(sentences):
    freq = Counter(p for o in sentences for p in o)
    vocab = [o for o, c in freq.items() if c > 1] # could use >2?
    vocab.insert(0, 'unk')
    string2idx = defaultdict(lambda: 0, {v: k for k,v in enumerate(vocab)})

    return vocab, string2idx
