"""
Taken from https://github.com/Bjarten/early-stopping-pytorch 
"""
__all__ = ['EarlyStopping']


class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.min_val_loss = None

    def __call__(self, val_loss, wrapper):
        score = -val_loss
        if self.min_val_loss is None:
            self.min_val_loss = val_loss
            if self.verbose:
                print(f'Validation loss decreased ({self.min_val_loss:.6f} --> {val_loss:.6f}).  storing state_dict...')
            wrapper.save_with_suffix()
        elif val_loss > self.min_val_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.min_val_loss = val_loss
            if self.verbose:
                print(f'Validation loss decreased ({self.min_val_loss:.6f} --> {val_loss:.6f}).  storing state_dict...')
            wrapper.save_with_suffix()
            self.counter = 0
