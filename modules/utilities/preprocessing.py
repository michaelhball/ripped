import gensim.downloader as api
import numpy as np
import spacy

from pathlib import Path


def tokenise(we_source, data):
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


class EmbeddingNode():
    def __init__(self, we, node):
        self.text = node.text.lower()
        self.dep = node.dep_
        self.embedding = we[self.text] if self.text in we else we['unk']
        self.children = [EmbeddingNode(we, c) for c in node.children]
