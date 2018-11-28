import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .linear_block import LinearBlock


class DependencyEncoder(nn.Module):
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
        self.params = nn.ParameterList([nn.Parameter(torch.zeros(self.embedding_dim, self.embedding_dim)) for _ in range(self.num_params)])
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(self.num_params)])
        self.evaluate = evaluate
        self._init_params()

        # to analyse dependency usage
        self.dep_freq = {}
        self.rare_dependencies = set()
    
    def _init_params(self):
        for param in self.params.parameters():
            nn.init.xavier_uniform_(param)
    
    def _is_leaf(self, node):
        return False if node.chidren else True

    def recur(self, node):
        x = Variable(torch.tensor([node.embedding], dtype=torch.float), requires_grad=True)
        if self._is_leaf(node):
            z = x
        else:
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
                D_c = self.params[dep_index] # dxd

                # # method 1
                # x_c = self.recur(c).repeat(self.embedding_dim, 1) # dxd
                # z_c = F.relu(x_c * D_c) # dxd # this perhaps isn't the best place to have activation.
                # z_cs.append(z_c)

                # method 2
                x_c = F.relu(self.recur(c).repeat(self.embedding_dim, 1)) # dxd
                z_c = x_c * D_c
                z_cs.append(z_c)

            if not z_cs:
                z = x
            else:
                zs = torch.stack(z_cs) # nxdxd (n = #children)
                mult, _ = torch.max(zs, 0) # dxd
                # mult = torch.mean(zs, 0) # dxd
                z = torch.matmul(x, mult) # 1xd for output to next layer

        return z

    def forward(self, input):
        with torch.set_grad_enabled(self.training):
            return self.recur(input)
