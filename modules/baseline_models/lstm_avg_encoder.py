import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAvgEncoder(nn.Module):
    def __init__(self, embedding_dim, num_layers):
        super().__init__()
        self.embedding_dim, self.num_layers = embedding_dim, num_layers
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers)

    def forward(self, x):
        seq_len = len(x)
        x = torch.tensor([x], requires_grad=True).view((seq_len, 1, self.embedding_dim))
        output, _ = self.lstm(x)
        
        return torch.mean(output, 0)
