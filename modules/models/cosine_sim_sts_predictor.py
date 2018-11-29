import torch.nn as nn
import torch.nn.functional as F


class CosineSimSTSPredictor(nn.Module):
    """
    A class to predict similarity based purely on the distance
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, input1, input2):
        x_1 = self.encoder(input1) # 1xd
        x_2 = self.encoder(input2) # 1xd
        cosine_sim = F.cosine_similarity(x_1, x_2)

        return cosine_sim