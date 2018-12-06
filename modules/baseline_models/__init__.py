from .linear_block import LinearBlock
from .pooling_predictor import PoolingPredictor


def create_pooling_baseline(layers, drops, pool_type):
    return PoolingPredictor(layers, drops, pool_type)
