from modules.utilities.imports_torch import *


class LinearBlock(nn.Module):
    """
    Packages a nn linear layer into a block with dropout.
    """
    def __init__(self, ni, nf, drop=0.5):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.lin(self.drop(x))
