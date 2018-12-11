import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_block import LinearBlock


class STSPredictor(nn.Module):
    """
    A class to predict similarity between two input sentences.
    """
    def __init__(self, encoder, layers, drops):
        super().__init__()
        self.encoder = encoder
        self.layers = nn.ModuleList([LinearBlock(layers[i], layers[i+1], drops[i]) for i in range(len(layers) - 1)])
    
    def forward(self, input1, input2):
        x1 = self.encoder(input1) # 1xd
        x2 = self.encoder(input2) # 1xd
        diff = (x1-x2).abs() # 1xd
        mult = x1 * x2 # 1xd
        x = torch.cat((diff, mult), 0).reshape(1, -1) # 1x4d
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)

        return l_x
