import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, layers, drops):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers)
        self.layers = nn.ModuleList([LinearBlock(layers[i], layers[i+1], drops[i]) for i in range(len(layers) - 1)])

    def forward(self, input):
        o1, hs1 = self.lstm(input[0])
        o2, hs2 = self.lstm(input[1])
        x1, x2 = hs1[-1], hs2[-1]
        
        x = torch.cat((x1, x2), 0).view(1, -1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)

        return l_x
