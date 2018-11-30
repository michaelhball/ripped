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
        x_1 = self.encoder(input1) # 1xd
        x_2 = self.encoder(input2) # 1xd
        # diff = (x_1-x_2).abs() # 1xd
        # mult = x_1 * x_2 # 1xd
        x = torch.cat((x_1, x_2), 0).reshape(1,-1) # 1x2d
        
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)

        return l_x
