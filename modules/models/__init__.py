from .basic_classifier import BasicClassifier
from .lstm_encoder import LSTMEncoder
from .pass_encoder import PassEncoder
from .pool_encoder import PoolEncoder
from .sts_predictor import STSPredictor

from torch.nn import Sequential

__all__ = ['create_encoder', 'create_sts_predictor', 'create_intent_classifier']

# def create_nli_predictor(embedding_dim, encoder_model, layers, drops, use_bias=False):
#     encoder = dependency_encoders[encoder_model](embedding_dim)
#     return NLIPredictor(encoder, layers, drops)

# def create_sts_predictor(embedding_dim, encoder_model, layers, drops, use_bias=False):
#     encoder = dependency_encoders[encoder_model](embedding_dim)
#     return STSPredictor(encoder, layers, drops)

# def create_multi_task_learner(embedding_dim, encoder_model, sts_dims, nli_dims):
#     if encoder_model == "pretrained":
#         encoder = PassEncoder()
#     else:
#         encoder = dependency_encoders[encoder_model](embedding_dim)
#     return MultiTaskLearner(encoder, sts_dims, nli_dims)


def create_classifier(layers, drops):
    return BasicClassifier(layers, drops)


def create_encoder(vocab, embedding_dim, encoder_model, *args):
    if encoder_model.startswith('pool'):
        encoder = PoolEncoder
    elif encoder_model == "lstm":
        encoder = LSTMEncoder
    elif encoder_model == "pass":
        encoder = PassEncoder
    else:
        print(f'encoder type: "{encoder_model}" not implemented')
    
    return encoder(vocab, embedding_dim, *args)


def create_sts_predictor(vocab, embedding_dim, encoder_model, predictor_model, layers, drops, *args):
    encoder = create_encoder(vocab, embedding_dim, encoder_model, *args)
    return STSPredictor(encoder, layers, drops)


def create_intent_classifier(vocab, embedding_dim, encoder_model, layers, drops, *args):
    encoder = create_encoder(vocab, embedding_dim, encoder_model, *args)
    classifier = BasicClassifier(layers, drops)
    return Sequential(encoder, classifier)
