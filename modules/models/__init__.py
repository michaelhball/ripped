from .basic_classifier import BasicClassifier
from .cosine_sim_sts_predictor import CosineSimSTSPredictor
from .dep_tree import DEPTree
from .infersent_encoder import InferSentEncoder
from .linear_block import LinearBlock
from .lstm_encoder import LSTMEncoder
from .multi_task_learner import MultiTaskLearner
from .pass_encoder import PassEncoder
from .pool_encoder import PoolEncoder
from .pos_lin import POSLin
from .pos_tree import POSTree
from .sts_predictor import STSPredictor


from torch.nn import Sequential


__all__ = ['create_encoder', 'create_sts_predictor', 'create_intent_classifier']


# dependency_encoders = {
#     "pos_lin": POSLin,
#     "pos_tree": POSTree,
#     "dep_tree": DEPTree,
# }

# predictors = {
#     "nn": STSPredictor,
#     "cosine": CosineSimSTSPredictor
# }

# def create_nli_predictor(embedding_dim, encoder_model, layers, drops, use_bias=False):
#     encoder = dependency_encoders[encoder_model](embedding_dim)
#     return NLIPredictor(encoder, layers, drops)

# def create_sts_predictor(embedding_dim, encoder_model, layers, drops, use_bias=False):
#     encoder = dependency_encoders[encoder_model](embedding_dim)
#     return STSPredictor(encoder, layers, drops)

# def create_sts_predictor_w_pretrained_encoder(layers, drops, predictor_type='nn', use_bias=False):
#     return predictors[predictor_type](PassEncoder(), layers, drops)
#     # return STSPredictor(PassEncoder(), layers, drops)

def create_multi_task_learner(embedding_dim, encoder_model, sts_dims, nli_dims):
    if encoder_model == "pretrained":
        encoder = PassEncoder()
    else:
        encoder = dependency_encoders[encoder_model](embedding_dim)
    return MultiTaskLearner(encoder, sts_dims, nli_dims)

def create_classifier(layers, drops):
    return BasicClassifier(layers, drops)



def create_encoder(vocab, embedding_dim, encoder_model, *args):
    if encoder_model.startswith('pool'):
        encoder = PoolEncoder
    elif encoder_model == "lstm":
        encoder = LSTMEncoder
    elif encoder_model.startswith("infersent"):
        encoder = InferSentEncoder
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
