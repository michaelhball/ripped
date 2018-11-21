from .basic_classifier import BasicClassifier
from .dependency_encoder import DependencyEncoder
from .linear_block import LinearBlock
from .pooling_linear_classifier import PoolingLinearClassifier
from .sentence_distance_sts_predictor import SentenceDistanceSTSPredictor
from .sts_predictor import STSPredictor


def create_sts_predictor(nlp, word_embeddings, embedding_dim, batch_size, dependency_map, layers, use_bias=False, drops=None):
    encoder = DependencyEncoder(nlp, word_embeddings, embedding_dim, batch_size, dependency_map, use_bias)

    return STSPredictor(encoder, layers, drops)


def create_sentence_distance_sts_predictor(nlp, word_embeddings, embedding_dim, batch_size, dependency_map, use_bias=False):
    encoder = DependencyEncoder(nlp, word_embeddings, embedding_dim, batch_size, dependency_map, use_bias)

    return SentenceDistanceSTSPredictor(encoder)