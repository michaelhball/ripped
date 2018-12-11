import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class POSTagEncoder(nn.Module):
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
        self.lstm = nn.LSTM(self.embedding_dim, self.embedding_dim, 1)
        self.evaluate = evaluate

        # # to analyse pos tag usage
        # self.tag_freq = {}
        # self.rare_tags = set()

    # def recur(self, node, node_reps):
    #     if node.pos not in self.tag_freq:
    #         self.tag_freq[node.pos] = 0
    #     else:
    #         self.tag_freq[node.pos] += 1
    #     if node.pos not in self.pos_tags:
    #         if node.pos not in self.rare_tags:
    #             self.rare_tags.add(node.pos)
    #         return

    #     x = Variable(torch.tensor([node.embedding], dtype=torch.float), requires_grad=True)
    #     D = self.params[self.pos_tags[node.pos]]
    #     z = D(x)
    #     node_reps.append(z)

    #     for c in node.chidren:
    #         self.recur(c, node_reps)

    def forward(self, input):
        with torch.set_grad_enabled(self.training):
            word_reps = []
            for (pos, emb) in input:
                x = Variable(torch.tensor([emb], dtype=torch.float), requires_grad=True)
                D = self.params[self.pos_tags[pos]]
                z = D(x)
                # z = F.relu(z)
                word_reps.append(z)

            z = torch.stack(word_reps)
            # output, _ = torch.max(z, 0)
            outputs, hidden_states = self.lstm(z)
            output, _ = torch.max(outputs, 0)

            return output
