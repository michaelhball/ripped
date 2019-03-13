import math
import numpy as np
import warnings

from modules.models import create_sts_predictor
from modules.utilities import euclid
from modules.utilities.imports import *
from modules.utilities.imports_torch import *


def get_sts_model(source):
    vocab = pickle.load(Path(f'./data/sts/{source}/pretrained/vocab.pkl').open('rb'))
    params = pickle.load(Path(f'./data/sts/{source}/pretrained/params.pkl').open('rb'))

    enc_params, pred_params = params['encoder'], params['predictor']
    emb_dim, hid_dim = enc_params['emb_dim'], enc_params['hid_dim']
    num_layers, output_type = enc_params['num_layers'], enc_params['output_type']
    bidir, fine_tune = enc_params['bidir'], enc_params['fine_tune']
    layers, drops = pred_params['layers'], pred_params['drops']

    enc_args = [hid_dim, num_layers, bidir, fine_tune, output_type]
    model = create_sts_predictor(vocab, emb_dim, 'lstm', 'mlp', layers, drops, *enc_args)
    model.load_state_dict(torch.load(f'./data/sts/{source}/pretrained/weights.pt', map_location=lambda storage, loc: storage))
    model.eval()
    
    return model


def sts_dist(model, u, v):
    u, v = torch.tensor([u]), torch.tensor([v])
    diff = (u-v).abs()
    mult = u * v
    x = torch.cat((diff, mult), 1)
    for l in model.layers:
        l_x = l(x)
        x = F.relu(l_x)

    sim = l_x.item()
    if sim <= 0:
        sim = 0.01

    return 1/(sim/5)


MODEL = None


class LabelProp():
    def __init__(self, x_l, y_l, x_u, nc, sim_measure='cosine', source=None):
        """
        Class for implementation of original label propagation algorithm.
        """
        if sim_measure == "sts":
            global MODEL
            if MODEL is None:
                MODEL = get_sts_model(source)
        self.sim_measure = sim_measure

        self.nl, self.nu, self.n = len(x_l), len(x_u), len(x_l)+len(x_u)
        self.nc = nc
        self.T = np.zeros((self.n,self.n), dtype=np.float)
        self.Y = np.zeros((self.n,self.nc), dtype=np.float)

        # initialise T
        self.ss = math.pow(self._calculate_sigma(x_l,y_l), 2)
        # print(self.ss)
        # self.ss = 0.007
        self.ss /= 3
        for i, u in enumerate(x_l):
            for j, v in enumerate(x_l):
                self.T[i,j] = self._lp_dist(u, v)
            for k, z in enumerate(x_u):
                dist = self._lp_dist(u, z)
                self.T[i,self.nl+k] = dist
                self.T[self.nl+k,i] = dist
        for i, u in enumerate(x_u):
            for j, v in enumerate(x_u):
                self.T[self.nl+i,self.nl+j] = self._lp_dist(u, v)

        self.T /= self.T.sum(axis=0)[np.newaxis,:] # column norm
        self.T /= self.T.sum(axis=1)[:,np.newaxis] # row norm

        # initialise Y
        self.Y = np.zeros((self.n,self.nc), dtype=np.float)
        for i, _ in enumerate(x_l):
            for j in range(self.nc):
                self.Y[i,j] = 1 if j == y_l[i] else 0
        self.Y_static = self.Y[:self.nl]
    
    def _dist_func(self, u, v):
        if self.sim_measure == "cosine":
            return euclid(u, v)
        elif self.sim_measure == 'sts':
            return sts_dist(MODEL, u, v)

    def _calculate_sigma(self, x_l, y_l):
        D = np.zeros((self.nl,self.nl), dtype=np.float)
        for i, u in enumerate(x_l):
            for j, v in enumerate(x_l):
                D[i,j] = self._dist_func(u, v)
        
        min_dist = float('inf')
        for i in range(self.nl):
            for j, dist in enumerate(D[i]):
                if y_l[i] != y_l[j] and dist != 0 and dist < min_dist:
                    min_dist = dist
            
        return min_dist / 3
    
    def _lp_dist(self, u, v):
        return np.exp(-(self._dist_func(u,v)/self.ss))
    
    def propagate(self, tol=0.0001, max_iter=10000):
        Y_prev = np.zeros((self.n,self.nc),dtype=np.float)
        for i in range(max_iter):
            if np.abs(self.Y-Y_prev).sum() < tol: break
            Y_prev = self.Y
            self.Y = np.matmul(self.T, self.Y) # Y <- TY
            # self.Y[self.nl:] /= self.Y[self.nl:].sum(axis=1)[:, np.newaxis] # row normalise Y_u
            self.Y[:self.nl] = self.Y_static # clamp labels
        else:
            warnings.warn(f'max_iter ({max_iter}) was reached without convergence')
    
    def recursive(self, x_l, y_l, x_u, y_u, tol=0.0001, max_iter=10000):
        x_indices = {str(x):i for i, x in enumerate(x_u)} # indices in orignial unlabeled set.
        xl, yl, xu, yu = x_l, y_l, x_u, y_u
        all_classifications, all_indices = [], []
        last_unlabeled_count = len(x_u)

        iters = 0
        while True:
            iters += 1
            self.__init__(xl, yl, xu, 21)
            self.propagate(tol, max_iter)
            classifications, indices = self.classify(threshold=True)

            # add correct indices to overall indices
            all_classifications += classifications
            for i in indices:
                all_indices.append(x_indices[str(xu[i])])

            # create new data for next round.
            for i in indices:
                xl = np.concatenate((xl, np.expand_dims(xu[i],axis=0)), axis=0)
                yl = np.append(yl, [yu[i]])
            xu = np.array([x for i,x in enumerate(xu) if i not in indices])
            yu = np.array([y for i,y in enumerate(yu) if i not in indices])

            # break condition
            if last_unlabeled_count - len(xu) == 0:
                break
            last_unlabeled_count = len(xu)
        
        return all_classifications, all_indices
    
    def p1nn(self, x_l, y_l, x_u):
        classifications = list(-1 * np.ones(self.nu, dtype=np.float))
        x_l_labels = {i:y_l[i] for i, _ in enumerate(x_l)}
        
        dist_u_to_l = np.zeros((self.nu,self.n), dtype=np.float)
        for i, u in enumerate(x_u):
            for j, v in enumerate(x_l):
                dist_u_to_l[i,j] = euclid(u,v)
            for k in range(self.nu):
                dist_u_to_l[i,self.nl+k] = float('inf')

        classified = set()
        while True:
            closest_row, closest_col = -1, -1
            closest_dist = float('inf')
            for j, row in enumerate(dist_u_to_l):
                if j not in classified:
                    dist = np.min(row)
                    if dist < closest_dist:
                        closest_row = j
                        closest_col = np.argmin(row)
                        closest_dist = dist
            
            if closest_row == -1:
                break

            classified.add(closest_row) # add to set showing which unlabeled examples we've classified
            label = x_l_labels[closest_col] # get label for unlabeled example
            classifications[closest_row] = label # add label to classification list for output
            x_l_labels[self.nl+closest_row] = label # add classification to dictionary of labeled examples.

            for j, row in enumerate(dist_u_to_l):
                if j not in classified:
                    dist_u_to_l[j,self.nl+closest_row] = euclid(x_u[j], x_u[closest_row])

        return classifications, [i for i in range(len(x_u))]

    def accuracy(self, y_u):
        preds, indices = self.classify(y_u, threshold=True)
        y_idxed = y_u[indices]
        print(f'accuracy: {np.sum(preds == y_idxed) / len(preds)} using {len(y_idxed)} unlabeled examples out of {self.nu}')
        preds = self.classify(y_u)
        return np.sum(preds == y_u) / len(preds)
    
    def classify(self, threshold=True):
        self.Y[self.nl:] /= self.Y[self.nl:].sum(axis=1)[:, np.newaxis] # normalise predictions
        indices, preds = [], []
        for i, row in enumerate(self.Y[self.nl:]):
            THRESH = 0.9 if threshold else 0
            # print(np.max(row))
            if np.max(row) >= THRESH:
                indices.append(i)
                preds.append(np.argmax(row))

        return preds, indices
