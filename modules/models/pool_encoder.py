import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolEncoder(nn.Module):
    """Encoder that pools pre-trained word embeddings. """
    def __init__(self, vocab, embedding_dim, pool_type='max'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=vocab.stoi['<pad>'])
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.embedding.weight.requires_grad = False # allowing fine-tuning
        self.pool_type = pool_type

    def pool(self, x, batch_size, is_max):
        """ Adaptive pooling utility. """
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(batch_size, -1)

    def forward(self, x):
        seq_len, batch_size = x.shape
        with torch.set_grad_enabled(self.training):
            emb = self.embedding(x)
            if self.pool_type == "max":
                output = self.pool(emb, batch_size, True)
            elif self.pool_type == "avg":
                output = self.pool(emb, batch_size, False)
            elif self.pool_type == "both":
                max_pool = self.pool(emb, batch_size, True)
                avg_pool = self.pool(emb, batch_size, False)
                output = torch.concat([max_pool, avg_pool], 1)
            
            return output
