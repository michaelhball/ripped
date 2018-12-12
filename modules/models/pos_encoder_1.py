import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class POSEncoder1(nn.Module):
    def __init__(self, embedding_dim, batch_size, pos_tags, evaluate=False):
        """
        Sentence embedding model using pos tags.
        Args:
            embedding_dim (int): dimension of word/phrase/sentence embeddings
            pos_tags (dict): map of pos tags to index (used for defining params)
            evaluate (bool): Indicator as to whether we want to evaluate embeddings.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_tags = pos_tags
        self.num_params = max(list(self.pos_tags.values())) + 1
        self.params = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.num_params)])
        self.evaluate = evaluate

    def forward(self, input):
        with torch.set_grad_enabled(self.training):
            word_reps = []
            for (pos, emb) in input:
                x = Variable(torch.tensor([emb], dtype=torch.float), requires_grad=True)
                D = self.params[self.pos_tags[pos]]
                z = F.relu(D(x))
                word_reps.append(z)

            z = torch.stack(word_reps)
            output, _ = torch.max(z, 0)

            return output
