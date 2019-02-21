import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_block import LinearBlock


class MultiTaskLearner(nn.Module):
    def __init__(self, encoder, sts_dims, nli_dims):
        """
        Multi-Task Learning Framework using STS and NLI.
        """
        super().__init__()
        self.encoder = encoder
        sts_layers, sts_drops = sts_dims
        nli_layers, nli_drops = nli_dims
        self.sts_head = nn.ModuleList([LinearBlock(sts_layers[i], sts_layers[i+1], sts_drops[i]) for i in range(len(sts_layers) - 1)])
        self.nli_head = nn.ModuleList([LinearBlock(nli_layers[i], nli_layers[i+1], nli_drops[i]) for i in range(len(nli_layers) - 1)])

    def forward(self, input_type, x1, x2): # x1 needs to be able to be a batch, and if so the encoder has to handle it.
        x1 = self.encoder(x1) # bs x d
        x2 = self.encoder(x2) # bs x d
        diff = (x1-x2).abs() # bs x d
        mult = x1 * x2 # bs x d
        x = torch.cat((diff, mult), 0).reshape(1, -1) # 1x2d # CHECK concatenation is on correct axis.
        
        # i think linear layers are automatically created to handle batching?
        if input_type == "sts":
            for l in self.sts_head:
                l_x = l(x)
                x = F.relu(l_x)
        elif input_type == "nli":
            for l in self.nli_head:
                l_x = l(x)
                x = F.relu(l_x)

        return l_x
