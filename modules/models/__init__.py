from .basic_classifier import BasicClassifier
from .cosine_sim_sts_predictor import CosineSimSTSPredictor
from .dep_tree import DEPTree
from .linear_block import LinearBlock
from .lstm_encoder import LSTMEncoder
from .multi_task_learner import MultiTaskLearner
from .pass_encoder import PassEncoder
from .pool_encoder import PoolEncoder
from .pos_lin import POSLin
from .pos_tree import POSTree
from .sts_predictor import STSPredictor


dependency_encoders = {
    "pos_lin": POSLin,
    "pos_tree": POSTree,
    "dep_tree": DEPTree,
}

predictors = {
    "nn": STSPredictor,
    "cosine": CosineSimSTSPredictor
}


def create_nli_predictor(embedding_dim, encoder_model, layers, drops, use_bias=False):
    encoder = dependency_encoders[encoder_model](embedding_dim)
    return NLIPredictor(encoder, layers, drops)


def create_sts_predictor(embedding_dim, encoder_model, layers, drops, use_bias=False):
    encoder = dependency_encoders[encoder_model](embedding_dim)
    return STSPredictor(encoder, layers, drops)


def create_sts_predictor_w_pretrained_encoder(layers, drops, predictor_type='nn', use_bias=False):
    return predictors[predictor_type](PassEncoder(), layers, drops)
    # return STSPredictor(PassEncoder(), layers, drops)


def create_multi_task_learner(embedding_dim, encoder_model, sts_dims, nli_dims):
    if encoder_model == "pretrained":
        encoder = PassEncoder()
    else:
        encoder = dependency_encoders[encoder_model](embedding_dim)
    return MultiTaskLearner(encoder, sts_dims, nli_dims)


def create_encoder(embedding_dim, encoder_model):
    return dependency_encoders[encoder_model](embedding_dim)


def create_intent_classifier(encoder_type, vocab, layers, drops, *args):
    from torch.nn import Sequential
    if encoder_type == "lstm":
        encoder = LSTMEncoder(vocab, *args)
    elif encoder_type == "we_pool":
        encoder = PoolEncoder(vocab, *args)
    else:
        print("UH OH")
    return Sequential(encoder, BasicClassifier(layers, drops))


def create_classifier(layers, drops):
    return BasicClassifier(layers, drops)
