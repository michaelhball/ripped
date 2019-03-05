from torchtext import data


__all__ = ['get_ic_data_iterator', 'get_ic_data_iterators', 'get_sts_data_iterators']


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
    train_di, val_di, test_di = MyIterator.splits(
        (train_ds, val_ds, test_ds),
        batch_sizes=batch_sizes,
        sort_key=lambda eg: (len(eg.x1), len(eg.x2)),
        repeat=False,
        batch_size_fn=sts_batch_size_fn,
        shuffle=True
    )

    return train_di, val_di, test_di


# code below modified from from http://nlp.seas.harvard.edu/2018/04/03/attention.html 
# for more efficient batching by sequence length - refer to https://goo.gl/ofLoN3 

global max_x_in_batch
def intent_batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_x_in_batch
    if count == 1:
        max_x_in_batch = 0
    max_x_in_batch = max(max_x_in_batch,  len(new.x))
    src_elements = count * max_x_in_batch
    return src_elements


global max_x1_in_batch, max_x2_in_batch
def sts_batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_x1_in_batch, max_x2_in_batch
    if count == 1:
        max_x1_in_batch = 0
        max_x2_in_batch = 0
    max_x1_in_batch = max(max_x1_in_batch, len(new.x1))
    max_x2_in_batch = max(max_x2_in_batch, len(new.x2))
    x1_elements = count * max_x1_in_batch
    x2_elements = count * max_x2_in_batch
    return max(x1_elements, x2_elements)


class MyIterator(data.Iterator):
    def __len__(self):
        return len(self.batches) # doesn't work for training data because batches is a generator
    
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))
