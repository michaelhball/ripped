import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrained_models.infersent.models import InferSent


class InferSentEncoder(nn.Module):
    """ Encoder using pretrained InferSent """
    def __init__(self, vocab, embedding_dim, bs, encoding_dim, pool_type, dropout, version, state_dict_path, w2v_path, sentences=None, fine_tune=False):
        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.fine_tune = fine_tune
        self.params_model = {
            'bsize': bs, 'word_emb_dim': embedding_dim, 'enc_lstm_dim': encoding_dim,
            'pool_type': pool_type, 'dpout_model': dropout, 'version': version
        }
        self.infersent = InferSent(self.params_model)
        self.infersent.load_state_dict(torch.load(state_dict_path))
        self.infersent.set_w2v_path(w2v_path)

        if sentences and not fine_tune:
            self.infersent.build_vocab(sentences, tokenize=False)
        else:
            pass
            # we need to fine-tune which means we need to do something with the vocab stuff... vocab input argument
            # needs to match with that created by InferSent's build_vocab
            # I know we can manually just set the w2vec attribute if i need to from vocab.Vectors stuff.

        # NB: Look inside the encode function of InferSEnt to work out how to implement
        # the data iteration/preprocessing we need to input sentences to this model.
        # Also, this seems very slow... and I'm not exactly sure why.

    def forward(self, x):
        if not self.fine_tune:
            return self.infersent.encode(x, tokenize=False)
        else:
            pass
