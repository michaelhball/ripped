from .basic_classifier import BasicClassifier
from .dep_tree import DEPTree
from .linear_block import LinearBlock
from .pos_lin import POSLin
from .pos_tree import POSTree
from .sts_predictor import STSPredictor


dependency_encoders = {
    "pos_lin": POSLin,
    "pos_tree": POSTree,
    "dep_tree": DEPTree,
}


def create_sts_predictor(embedding_dim, encoder_model, layers, drops, use_bias=False):
    encoder = dependency_encoders[encoder_model](embedding_dim)
    return STSPredictor(encoder, layers, drops)


def create_encoder(embedding_dim, encoder_model):
    return dependency_encoders[encoder_model](embedding_dim)


def create_classifier(layers, drops):
    return BasicClassifier(layers, drops)
