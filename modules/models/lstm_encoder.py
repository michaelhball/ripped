import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class LSTMEncoder(nn.Module):
    def __init__(self, vocab, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.batch_size = 1
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = False # EXPERIMENT
        self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=vocab.stoi['<pad>'])
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.embedding.weight.requires_grad = False # EXPERIMENT
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=self.bidirectional)

    def one_hidden(self):
        """
        Resets one hidden layer
        """
        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        return Variable(self.weights.new(num_layers, self.batch_size, self.hidden_dim).zero_())

    def reset(self):
        """
        Resets the networks hidden layers for the next iteration of training.
        """
        self.weights = next(self.parameters()).data
        self.hidden = (self.one_hidden(), self.one_hidden())
    
    def repackage_hidden(self, h):
        """
        Repackages a variable to allow it to forget its history.
        """
        return h.detach() if type(h) == torch.Tensor else tuple(self.repackage_hidden(v) for v in h)

    def pool(self, x, batch_size, is_max):
        """
        """
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(batch_size, -1)

    def forward(self, x):
        seq_len, batch_size = x.shape
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.reset()

        with torch.set_grad_enabled(self.training):
            emb = self.embedding(x)
            lstm_out, self.hidden = self.lstm(emb, self.hidden) # (sl, bs, emb_dim) (emb_dim*2 if bidirectional)
            max_pool = self.pool(lstm_out, self.batch_size, True) # bs,emb_dim
            # avg_pool = self.pool(lstm_out, self.batch_size, False) # bs,emb_dim
            output = max_pool

            self.hidden = self.repackage_hidden(self.hidden)
            
            return output
