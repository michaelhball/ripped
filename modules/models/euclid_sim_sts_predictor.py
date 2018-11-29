import torch.nn as nn
import torch.nn.functional as F


class EuclidSimSTSPredictor(nn.Module):
    """
    A class to predict similarity based purely on the distance
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, input1, input2):
        x_1 = F.normalize(self.encoder(input1), p=2, dim=1) # 1xd
        x_2 = F.normalize(self.encoder(input2), p=2, dim=1) # 1xd
        euclid_dist = F.normalize(x_1-x_2) # 1x1
        euclid_sim = 1 / (1 + euclid_dist)

        return euclid_sim