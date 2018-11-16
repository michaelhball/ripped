import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_block import LinearBlock


class STSPredictor(nn.Module):
    """
    A class to predict similarity between two input sentences.
    """
    def __init__(self, encoder, layers, drops=None):
        super().__init__()
        self.encoder = encoder
        if drops:
            self.layers = nn.ModuleList([LinearBlock(layers[i], layers[i+1], drops[i]) for i in range(len(layers) - 1)])
        else:
            self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
    
    def forward(self, input1, input2): # this isn't set up for batching yet
        x_1 = self.encoder(input1) # 1xd
        x_2 = self.encoder(input2) # 1xd
        diff = (x_1-x_2).abs() # 1xd
        mult = x_1 * x_2 # 1xd

        x = torch.cat((x_1, x_2, diff, mult), 0).reshape(1,-1) # 1x4d
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)

        return l_x, x # 1x1


# if __name__ == "__main__":
#     import gensim.downloader as api
#     import spacy
#     from dependency_encoder import DependencyEncoder

#     DEPENDENCIES = {
#         'advcl': 0, 'advmod': 0, 'obl': 0, 'acl': 1, 'amod': 1, 'nmod': 1, 'nummod': 1,
#         'compound': 2, 'fixed': 2, 'flat': 2, 'goeswith': 2, 'csubj': 3, 'nsubj': 3,
#         'appos': 4, 'aux': 5, 'case': 6, 'cc': 7, 'ccomp': 8, 'conj': 9, 'cop': 10,
#         'dep': 11, 'det': 12, 'iobj': 13, 'list': 14, 'mark': 15, 'obj': 16, 'orphan': 17,
#         'parataxis': 18, 'vocative': 19, 'xcomp': 20
#     }

#     we = api.load('glove-wiki-gigaword-50')
#     nlp = spacy.load('en')
#     enc = DependencyEncoder(nlp, we, 50, 1, DEPENDENCIES)
#     layers = [200, 50, 1] # totally random at this point
#     pred = STSPredictor(enc, layers)
#     # s1 = "the young boys were playing outside."
#     # s2 = "the old men were sitting inside."
#     # l_x, x = pred(s1, s2)
#     print(pred.encoder.state_dict().keys())
#     print(pred.parameters())