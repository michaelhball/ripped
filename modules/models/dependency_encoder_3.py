import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

# from modules.utilities import EmbeddingNode

from .linear_block import LinearBlock

class DependencyEncoder3(nn.Module):
    def __init__(self, embedding_dim, batch_size, dependency_map, evaluate=False):
        """
        Sentence embedding model using dependency parse structure.
        Args:
            embedding_dim (int): dimension of word/phrase/sentence embeddings
            batch_size (int): training batch size (1 until I work out how to parallelise)
            dependency_map (dict): map of dependency labels to index (used for defining params)
            evaluate (bool): Indicator as to whether we want to evaluate embeddings.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.dependency_map = dependency_map
        self.num_params = max(list(self.dependency_map.values())) + 1
        self.params = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_params)])
        self.lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, 1)
        self.evaluate = evaluate

        # to analyse dependency usage
        self.dep_freq = {}
        self.rare_dependencies = set()
    
    def _is_leaf(self, node):
        return False if node.chidren else True

    def recur(self, node):
        x = Variable(torch.tensor([node.embedding], dtype=torch.float), requires_grad=True)
        z_cs = []
        for c in node.chidren:
            if c.dep not in self.dep_freq:
                self.dep_freq[c.dep] = 0
            else:
                self.dep_freq[c.dep] += 1
            if c.dep not in self.dependency_map:
                if c.dep not in self.rare_dependencies:
                    self.rare_dependencies.add(c.dep)
                continue
            
            dep_index = self.dependency_map[c.dep]
            D_c = self.params[dep_index]
            x_c = self.recur(c)
            z_c = D_c(x_c)
            z_cs.append(z_c)

        zs = torch.stack(z_cs + [x]) # nx1xd (n = #children+1)
        output, hidden_state = self.lstm(zs)
        z = hidden_state[-1].view(1, -1) # just using final hidden state

        return z

    def forward(self, input):
        if self.evaluate: # need to be able to pass this down and add representation at every node.
            input.representation = None
        with torch.set_grad_enabled(self.training):
            return self.recur(input)


class RepresentationTree():
    def __init__(self, embedding_tree):
        self.representation = None
        self.text = embedding_tree.tree
        self.dep = embedding_tree.dep
        self.embedding = embedding_tree.embedding
        self.children = [RepresentationTree(c) for c in embedding_tree.children]
