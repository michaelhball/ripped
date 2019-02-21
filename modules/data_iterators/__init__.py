from .easy_iterator import EasyIterator
from .nli_data_iterator import NLIDataIterator
from .senteval_data_iterator import SentEvalDataIterator
from .sts_data_iterator import STSDataIterator


import torch
from torchtext import data


def get_intent_classification_data_iterators(train_ds, val_ds, test_ds, batch_sizes):
    train_di, val_di, test_di = data.BucketIterator.splits(
        (train_ds, val_ds, test_ds),
        batch_sizes = batch_sizes,
        sort_key = lambda x: len(x.x),
        sort_within_batch = False,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    return train_di, val_di, test_di