import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .linear_block import LinearBlock


class DEPEncoder1(nn.Module):
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
        self.evaluate = evaluate

    def recur(self, node):
        x = Variable(torch.tensor([node.embedding], dtype=torch.float), requires_grad=True)
        z_cs = []
        for c in node.chidren:
            x_c = self.recur(c)
            D_c = self.params[self.dependency_map[c.dep]]
            z_cs.append(F.relu(D_c(x_c)))
        
        zs = torch.stack(z_cs + [x])
        z, _ = torch.max(zs, 0)

        # if self.evaluate:
        #     node.representation = z.detach().numpy()
        #     _, hs = self.lstm(torch.stack([x]))
        #     node.embedding = hs[-1].view(1, -1).detach().numpy()

        return z

    def forward(self, input):
        with torch.set_grad_enabled(self.training):
            output = self.recur(input)
            if self.evaluate:
                return input
            return output
