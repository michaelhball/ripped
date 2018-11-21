import pickle
import spacy
import sys

from pathlib import Path


class Node():
    def __init__(self, token):
        self.token = token
        self.doc = token.doc # spacy document from which this token came
        self.sent = token.sent # sentence from which this token came
        self.root = self.sent.root # root of the sentence from which this token came
        self.text = token.text # token text
        self.parent = token.head # parent node
        self.dep = token.dep_ # dependency relation to parent
        self.ent_type = token.ent_type_ # named entity type (could start using these for names??)
        self.children = [Node(c) for c in token.children]
    
    def _string(self, indent):
        retval = '---[{0}]---> {1}\n'.format(self.dep, self.text)
        tabs = ''
        for i in range(indent):
            tabs += '\t'
        if self.children:
            for child in self.children:
                retval += tabs + child.string(indent+1)
        
        return retval

    def __str__(self):
        return self._string(1)

    def is_root(self):
        return True if self.dep == 'root' else False
    
    def node_phrase(self):
        return self.token.subtree


class EmbeddingNode():
    def __init__(self, node):
        self.text = node.text
        self.dep = node.dep
        self.embedding = None # NEED TO GET FROM WORD EMBEDDINGS
        self.chidren = [EmbeddingNode(c) for c in node.children]
    
    # WE NEED TO CONVERT THIS INTO A LINEAR REPRESENTATION, AND THEN MODiFY FORWARD METHOD OF DEPENDENCY ENCODER RUN ON THESE LINEAR METHODS...
    # NB: SPACY ALREADY RETURNS THE PARSE LINEARLY, SO I JUST NEED AN ORDERING OF NODES TO VISIT?
    # SEEMS tricky, largely due to the possibility of multiple children per node.


if __name__ == "__main__":
    sentence = "The young boys are playing outside"
    nlp = spacy.load('en')
    root = list(nlp(sentence).sents)[0].root
    root_node = Node(root)
    print(root_node)
