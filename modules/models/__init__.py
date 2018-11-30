from .basic_classifier import BasicClassifier
from .cosine_sim_sts_predictor import CosineSimSTSPredictor
from .dependency_encoder_1 import DependencyEncoder1
from .dependency_encoder_2 import DependencyEncoder2
from .dependency_encoder_3 import DependencyEncoder3
from .euclid_sim_sts_predictor import EuclidSimSTSPredictor
from .linear_block import LinearBlock
from .pooling_linear_classifier import PoolingLinearClassifier
from .sts_predictor import STSPredictor


dependency_encoders = {
    "1": DependencyEncoder1,
    "2": DependencyEncoder2,
    "3": DependencyEncoder3
}


def create_sts_predictor(embedding_dim, batch_size, dependency_map, encoder_model, layers, drops, use_bias=False):
    encoder = dependency_encoders[encoder_model](embedding_dim, batch_size, dependency_map)
    return STSPredictor(encoder, layers, drops)


def create_cosine_sim_sts_predictor(embedding_dim, batch_size, dependency_map, encoder_model):
    encoder = dependency_encoders[encoder_model](embedding_dim, batch_size, dependency_map)
    return CosineSimSTSPredictor(encoder)


def create_euclid_sim_sts_predictor(embedding_dim, batch_size, dependency_map, encoder_model):
    encoder = dependency_encoders[encoder_model](embedding_dim, batch_size, dependency_map)
    return EuclidSimSTSPredictor(encoder)
