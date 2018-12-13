import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from modules.utilities import my_dependencies, universal_tags
s

class DEPTree(nn.Module):
    def __init__(self, embedding_dim, evaluate=False):
        """
        Sentence embedding model using dependency parse structure.
        Args:
            embedding_dim (int): dimension of word/phrase/sentence embeddings
            evaluate (bool): Indicator as to whether we want to evaluate embeddings.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.evaluate = evaluate
        self.dep_tags = my_dependencies
        self.pos_tags = universal_tags
        self.num_dep_params = max(list(self.dep_tags.values())) + 1
        self.num_pos_params = max(list(self.pos_tags.values())) + 1
        self.dep_params = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_dep_params)])
        self.pos_params = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_pos_params)])

        self.dep_freq = {}

    def recur(self, node):
        x = Variable(torch.tensor([node.embedding], dtype=torch.float), requires_grad=True)
        x = F.relu(self.pos_params[self.pos_tags[node.pos]](x))
        z_cs = []
        for c in node.chidren:
            if c.dep not in self.dep_tags:
                continue
            x_c = self.recur(c)
            D_c = self.dep_params[self.dep_tags[c.dep]]
            z_c = F.relu(D_c(x_c))
            z_cs.append(z_c)
        
        z, _ = torch.max(torch.stack(z_cs+[x]), 0)
        return z

    def forward(self, input):
        with torch.set_grad_enabled(self.training):
            output = self.recur(input)
            if self.evaluate:
                return input
            return output
