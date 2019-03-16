from .infersent_sts_iterator import InferSentIterator

from torchtext import data

__all__ = ['get_ic_data_iterator', 'get_ic_data_iterators', 'get_sts_data_iterators', 'InferSentIterator']


def get_ic_data_iterator(ds, batch_size):
    return data.BucketIterator(
        ds,
        batch_size=batch_size,
        sort_key=lambda eg: len(eg.x),
        repeat=False,
        shuffle=True
    )


def get_ic_data_iterators(train_ds, val_ds, test_ds, batch_sizes):
    train_di, val_di, test_di = data.BucketIterator.splits(
        (train_ds, val_ds, test_ds),
        batch_sizes=batch_sizes,
        sort_key=lambda eg: len(eg.x),
        repeat=False,
        shuffle=True
    )
    
    return train_di, val_di, test_di


def get_sts_data_iterators(train_ds, val_ds, test_ds, batch_sizes):
    train_di, val_di, test_di = data.BucketIterator.splits(
        (train_ds, val_ds, test_ds),
        batch_sizes=batch_sizes,
        sort_key=lambda eg: (len(eg.x1), len(eg.x2)),
        repeat=False,
        shuffle=True
    )

    return train_di, val_di, test_di
