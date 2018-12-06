import torch.nn as nn
import torch.nn.functional as F

from .linear_block import LinearBlock


class BasicClassifier(nn.Module):
    """
    A basic linear classifier that performs affine operations + nonlinearity
        in a feedforward MLP according to the given layer dimensions
        specifications.
    """
    def __init__(self, layers, drops=None):
        super().__init__()
        self.layers = nn.ModuleList([LinearBlock(layers[i], layers[i+1], drops[i]) for i in range(len(layers) - 1)])

    def forward(self, x):
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        
        return l_x