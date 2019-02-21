import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class PoolEncoder(nn.Module):
    def __init__(self, vocab, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=vocab.stoi['<pad>'])
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.embedding.weight.requires_grad = False

    def pool(self, x, batch_size, is_max):
        """
        """
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(batch_size, -1)

    def forward(self, x):
        seq_len, self.batch_size = x.shape
        with torch.set_grad_enabled(self.training):
            emb = self.embedding(x)
            max_pool = self.pool(emb, self.batch_size, True) # bs,emb_dim
            
            return max_pool
