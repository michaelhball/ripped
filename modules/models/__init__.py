from .basic_classifier import BasicClassifier
from .dependency_encoder import DependencyEncoder
from .linear_block import LinearBlock
from .pooling_linear_classifier import PoolingLinearClassifier
from .cosine_dist_sts_predictor import CosineDistSTSPredictor
from .sts_predictor import STSPredictor


def create_sts_predictor(nlp, word_embeddings, embedding_dim, batch_size, dependency_map, layers, use_bias=False, drops=None):
    encoder = DependencyEncoder(nlp, word_embeddings, embedding_dim, batch_size, dependency_map, use_bias)

    return STSPredictor(encoder, layers, drops)


def create_cosine_dist_sts_predictor(embedding_dim, batch_size, dependency_map):
    encoder = DependencyEncoder(embedding_dim, batch_size, dependency_map)

    return CosineDistSTSPredictor(encoder)