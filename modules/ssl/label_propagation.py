import math
import numpy as np
import warnings

from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from modules.utilities import euclid


def label_prop_classify(xs_l, ys_l, xs_u, type_, pd, return_confidences=False):
    """
    Function to perform label propagation algorithm.
    Args:
        xs_l (np.ndarray): embeddings for labeled data
        ys_l (np.ndarray): labels for labeled data
        xs_u (np.ndarray): embeddings for unlabeled data
        ys_u (np.ndarray): labels for unlabeled data (not shown on purpose)
        type_ (str): 'propagation'|'spreading'
        pd (dict): param_dict -- keys must be 'gamma' (if rbf), 'n_neighbors' (if knn),
                   'alpha' (if spreading), 'max_iter', and 'tol'
    Returns:
        Classifications for the unlabeled data and accuracy of these classifications.
    """
    label_prop_model = LabelSpreading(**pd) if type_ == "spreading" else LabelPropagation(**pd)
    x_input = np.append(xs_l, xs_u, axis=0)
    y_input = np.append(ys_l, [-1 for _ in range(len(xs_u))], axis=0)
    label_prop_model.fit(x_input, y_input)
    # label_prop_model.fit(xs_l, ys_l)
    if return_confidences:
        pass
    else:
        classifications = label_prop_model.predict(xs_u)

    return classifications


class LabelProp():
    def __init__(self, x_l, y_l, x_u, nc, sigma=0.05):
        """
        Class for implementation of original label propagation algorithm.
        """
        self.nl, self.nu, self.n = len(x_l), len(x_u), len(x_l)+len(x_u)
        self.nc = nc
        self.T = np.zeros((self.n,self.n), dtype=np.float)
        self.Y = np.zeros((self.n,self.nc), dtype=np.float)

        # initialise T
        ss = math.pow(sigma, 2)
        for i, u in enumerate(x_l):
            for j, v in enumerate(x_l):
                self.T[i,j] = self._lp_dist(u, v, ss)
            for k, z in enumerate(x_u):
                dist = self._lp_dist(u, z, ss)
                self.T[i,self.nl+k] = dist
                self.T[self.nl+k,i] = dist
        for i, u in enumerate(x_u):
            for j, v in enumerate(x_u):
                self.T[self.nl+i,self.nl+j] = self._lp_dist(u, v, ss)

        self.T /= self.T.sum(axis=0)[np.newaxis,:] # column norm
        self.T /= self.T.sum(axis=1)[:,np.newaxis] # row norm

        # initialise Y
        self.Y = np.zeros((self.n,self.nc), dtype=np.float)
        for i, _ in enumerate(x_l):
            for j in range(self.nc):
                self.Y[i,j] = 1 if j == y_l[i] else 0
        for i in range(self.nu):
            self.Y[self.nl+i] = 0
        self.Y_static = self.Y[:self.nl]
    
    def _lp_dist(self, u, v, ss):
        return np.exp(-(euclid(u,v)/ss))
    
    def propagate(self, tol=0.0001, max_iter=10000, verbose=False):
        if verbose:
            print('beginning label propagation algorithm')

        Y_prev = np.zeros((self.n,self.nc),dtype=np.float)
        for i in range(max_iter):
            if np.abs(self.Y-Y_prev).sum() < tol: break
            Y_prev = self.Y
            self.Y = np.matmul(self.T, self.Y) # Y <- TY
            self.Y[:self.nl] = self.Y_static # clamp labels
        else:
            warnings.warn(f'max_iter ({max_iter}) was reached without convergence')
        
        if verbose:
            print(f'completed propagation in {i} iterations')
        
        self.Y /= self.Y.sum(axis=1)[:, np.newaxis] # normalise predictions

    def accuracy(self, y_u):
        preds, indices = self.classify(y_u, threshold=True)
        y_idxed = y_u[indices]
        print(f'accuracy: {np.sum(preds == y_idxed) / len(preds)} using {len(y_idxed)} unlabeled examples out of {self.nu}')

        preds = self.classify(y_u)
        return np.sum(preds == y_u) / len(preds)
    
    def classify(self, y_u, threshold=False):
        if threshold:
            indices, preds = [], []
            for i, row in enumerate(self.Y[self.nl:]):
                if np.max(row) == 1: # i.e. if the algorithm is 'certain'
                    indices.append(i)
                    preds.append(np.argmax(row))
            return preds, indices

        return np.argmax(self.Y[self.nl:], axis=1)
