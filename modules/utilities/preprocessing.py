import gensim.downloader as api
import numpy as np
import spacy

from collections import Counter, defaultdict
from pathlib import Path

from .tree import EmbeddingNode


def convert(embeddings, node):
    # to convert from EmbeddingTree to unordered list of embeddings for testing pooling baseline.
    embeddings.append(node.embedding)
    for c in node.chidren:
        convert(embeddings, c)


def tokenise_sent_tree(we, nlp, sentence):
    return EmbeddingNode(we, list(nlp(sentence).sents)[0].root)


def tokenise_sent_og(we, nlp, sentence):
    return [(t.pos_, we[t.text] if t.text in we else we['unk']) for t in nlp(str(sentence))]


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


def tokenise_sent(nlp, s):
        return [t.text.lower() for t in nlp(str(s))]


def create_vocab(sentences):
    freq = Counter(p for o in sentences for p in o)
    vocab = [o for o, c in freq.items() if c > 1] # could use >2?
    vocab.insert(0, 'unk')
    string2idx = defaultdict(lambda: 0, {v: k for k,v in enumerate(vocab)})

    return vocab, string2idx
