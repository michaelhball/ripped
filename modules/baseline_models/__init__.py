from .linear_block import LinearBlock
from .lstm_encoder import LSTMEncoder
from .lstm_avg_encoder import LSTMAvgEncoder
from .lstm_max_encoder import LSTMMaxEncoder
from .lstm_predictor import LSTMPredictor
from .pooling_predictor import PoolingPredictor


def create_pooling_baseline(layers, drops, pool_type):
    return PoolingPredictor(layers, drops, pool_type)


def create_lstm_baseline(embedding_dim, num_layers, layers, drops):
    encoder = LSTMEncoder(embedding_dim, num_layers)
    return LSTMPredictor(encoder, layers, drops)


def create_lstm_avg_baseline(embedding_dim, num_layers, layers, drops):
    encoder = LSTMAvgEncoder(embedding_dim, num_layers)
    return LSTMPredictor(encoder, layers, drops)


def create_lstm_max_baseline(embedding_dim, num_layers, layers, drops):
    encoder = LSTMMaxEncoder(embedding_dim, num_layers)
    return LSTMPredictor(encoder, layers, drops)
