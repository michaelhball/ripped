import torch.nn as nn
import torch.nn.functional as F


class CosineSimSTSPredictor(nn.Module):
    """
    A class to predict similarity based purely on the cosine sim
    """
    def __init__(self, encoder, layers=None, drops=None):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, x1, x2):
        return F.cosine_similarity(self.encoder(x1), self.encoder(x2))
