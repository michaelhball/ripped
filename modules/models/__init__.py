from .basic_classifier import BasicClassifier
from .cosine_sim_sts_predictor import CosineSimSTSPredictor
from .dependency_encoder import DependencyEncoder
from .dependency_encoder_2 import DependencyEncoder2
from .dependency_encoder_3 import DependencyEncoder3
from .euclid_sim_sts_predictor import EuclidSimSTSPredictor
from .linear_block import LinearBlock
from .pooling_linear_classifier import PoolingLinearClassifier
from .sts_predictor import STSPredictor


def create_sts_predictor(nlp, word_embeddings, embedding_dim, batch_size, dependency_map, layers, use_bias=False, drops=None):
    encoder = DependencyEncoder(nlp, word_embeddings, embedding_dim, batch_size, dependency_map, use_bias)

    return STSPredictor(encoder, layers, drops)


def create_cosine_sim_sts_predictor(embedding_dim, batch_size, dependency_map, encoder_model=1):
    if encoder_model == 1:
        encoder = DependencyEncoder(embedding_dim, batch_size, dependency_map)
    elif encoder_model == 2:
        encoder = DependencyEncoder2(embedding_dim, batch_size, dependency_map)
    elif encoder_model == 3:
        encoder = DependencyEncoder3(embedding_dim, batch_size, dependency_map)

    return CosineSimSTSPredictor(encoder)


def create_euclid_sim_sts_predictor(embedding_dim, batch_size, dependency_map):
    encoder = DependencyEncoder(embedding_dim, batch_size, dependency_map)

    return EuclidSimSTSPredictor(encoder)