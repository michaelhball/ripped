import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_block import LinearBlock


class LSTMPredictor(nn.Module):
    def __init__(self, encoder, layers, drops):
        super().__init__()
        self.encoder = encoder
        self.layers = nn.ModuleList([LinearBlock(layers[i], layers[i+1], drops[i]) for i in range(len(layers) - 1)])

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        diff = (x1 - x2).abs()
        mult = x1 * x2
        x = torch.cat((diff, mult), 0).view(1, -1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)

        return l_x
