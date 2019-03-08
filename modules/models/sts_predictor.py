from modules.utilities.imports_torch import *

from .linear_block import LinearBlock


class STSPredictor(nn.Module):
    """
    A class to predict similarity between two input sentences.
    """
    def __init__(self, encoder, layers, drops):
        super().__init__()
        self.encoder = encoder
        self.layers = nn.ModuleList([LinearBlock(layers[i], layers[i+1], drops[i]) for i in range(len(layers) - 1)])

    def pool(self, x, batch_size, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(batch_size, -1)

    def forward(self, x1, x2):
        x1, x2 = self.encoder(x1), self.encoder(x2)
        diff = (x1-x2).abs()
        mult = x1 * x2
        x = torch.cat((diff, mult), 1) # bs x (2xd)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)

        return l_x
