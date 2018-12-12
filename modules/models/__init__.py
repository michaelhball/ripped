from .basic_classifier import BasicClassifier
from .cosine_sim_sts_predictor import CosineSimSTSPredictor
from .dep_encoder_1 import DEPEncoder1
from .dep_encoder_1_lstm import DEPEncoder1LSTM
from .euclid_sim_sts_predictor import EuclidSimSTSPredictor
from .linear_block import LinearBlock
from .pos_encoder_1 import POSEncoder1
from .pos_encoder_1_lstm import POSEncoder1LSTM
from .pos_encoder_2 import POSEncoder2
from .pos_encoder_2_lstm import POSEncoder2LSTM
from .sts_predictor import STSPredictor


dependency_encoders = {
    "pos1": POSEncoder1,
    "pos1_lstm": POSEncoder1LSTM,
    "pos2": POSEncoder2,
    "pos2_lstm": POSEncoder2LSTM,
    "dep1": DEPEncoder1,
    "dep1_lstm": DEPEncoder1LSTM
}


def create_sts_predictor(embedding_dim, batch_size, param_map, encoder_model, layers, drops, use_bias=False):
    encoder = dependency_encoders[encoder_model](embedding_dim, batch_size, param_map)
    return STSPredictor(encoder, layers, drops)


def create_cosine_sim_sts_predictor(embedding_dim, batch_size, dependency_map, encoder_model):
    encoder = dependency_encoders[encoder_model](embedding_dim, batch_size, dependency_map)
    return CosineSimSTSPredictor(encoder)


def create_euclid_sim_sts_predictor(embedding_dim, batch_size, dependency_map, encoder_model):
    encoder = dependency_encoders[encoder_model](embedding_dim, batch_size, dependency_map)
    return EuclidSimSTSPredictor(encoder)


def create_encoder(embedding_dim, batch_size, dependency_map, encoder_model):
    return dependency_encoders[encoder_model](embedding_dim, batch_size, dependency_map)


def create_classifier(layers, drops):
    return BasicClassifier(layers, drops)
