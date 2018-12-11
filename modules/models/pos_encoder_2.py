import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .linear_block import LinearBlock


class POSEncoder2(nn.Module):
    def __init__(self, embedding_dim, batch_size, pos_tags, evaluate=False):
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
        self.pos_tags = pos_tags
        self.num_params = max(list(self.pos_tags.values())) + 1
        self.params = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_params)])
        self.lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, 1)
        self.evaluate = evaluate

    def recur(self, node):
        x = Variable(torch.tensor([node.embedding], dtype=torch.float), requires_grad=True)
        z_cs = []
        for c in node.chidren:
            D_c = self.params[self.pos_tags[c.pos]]
            x_c = self.recur(c)
            z_c = D_c(x_c)
            z_cs.append(z_c)

        zs = torch.stack(z_cs + [x]) # nx1xd (n = #children+1)
        outputs, hidden_state = self.lstm(zs)
        z, _ = torch.max(outputs, 0)

        if self.evaluate:
            node.representation = z.detach().numpy()
            outputs, _ = self.lstm(torch.stack([x]))
            i, _ = torch.max(outputs, 0)
            node.embedding = i.detach().numpy()

        return z

    def forward(self, input):
        with torch.set_grad_enabled(self.training):
            output = self.recur(input)
            if self.evaluate:
                return input
            return output
