import matplotlib.pyplot as plt
import warnings

from scipy import spatial
from scipy.stats import entropy

from modules.utilities.imports import *
from modules.utilities.imports_torch import *


class LabelProp():
    """
    Class for implementation of label propagation algorithm.
    Args:
        x_l (list(emb)): labeled data in embedded space
        y_l (list(int)): labels for labeled data
        x_u (list(emb)): unlabeled data in embedded space
        data_source (str): dataset
        display (bool): whether to print progress (for larger datasets).
    """
    def __init__(self, x_l, y_l, x_u, y_u, nc, data_source='chatbot', epsilon=0, sigma=None):
        st = time.time()
        self.data_source = data_source
        self.nl, self.nu, self.n, self.nc = len(x_l), len(x_u), len(x_l)+len(x_u), nc

        # calculate sigma
        self.sigma, T_ll = self._mst_heuristic_sigma(x_l, y_l)
        if sigma is not None:
            self.sigma = sigma
        self.ss = math.pow(self.sigma, 2)

        # initialise T
        self.T = np.zeros((self.n, self.n), dtype=np.float)
        self.T[:self.nl,:self.nl] = T_ll
        T_lu = spatial.distance.cdist(x_l, x_u)
        self.T[:self.nl,self.nl:] = T_lu
        T_ul = np.transpose(T_lu)
        self.T[self.nl:,:self.nl] = T_ul
        T_uu = spatial.distance.cdist(x_u, x_u)
        self.T[self.nl:,self.nl:] = T_uu
        self.T = np.exp(-self.T/self.ss) # weighting function

        # epsilon-interpolation smoothing w uniform transition matrix
        U = float(1/self.n) * np.ones((self.n, self.n), dtype=np.float)
        self.T = epsilon * U + (1 - epsilon) * self.T

        # normalise
        self.T /= self.T.sum(axis=0)[np.newaxis,:] # column norm
        self.T /= self.T.sum(axis=1)[:,np.newaxis] # row norm

        # initialise Y
        self.Y = np.zeros((self.n, self.nc), dtype=np.float)
        for i, _ in enumerate(x_l):
            for j in range(self.nc):
                self.Y[i,j] = 1 if j == y_l[i] else 0
        self.Y_static = self.Y[:self.nl]

    def _mst_heuristic_sigma(self, x_l, y_l):
        """
        Calculates sigma according to MST for the graph
            defined by label data. Output is the minimum
            distance between two points in different classes
            divided by 3 (using 3sigma rule of normal distribution).
        Args:
            x_l (list): encoded sentences from labeled data
            y_l (list(float)): corresponding labels
        Returns:
            The sigma value calculated according to this heuristic.
        """
        D = spatial.distance.cdist(x_l, x_l)
        min_dist = float('inf')
        for i in range(self.nl):
            for j, dist in enumerate(D[i]):
                if y_l[i] != y_l[j] and dist > 0 and dist < min_dist:
                    min_dist = dist
            
        return min_dist / 3 , D

    def propagate(self, tol=0.0001, max_iter=10000, data_for_plotting=None):
        """
        Performs label propagation until convergence
            (absolute difference in label matrix).
        Args:
            tol (float): tolerance defining convergence point
            max_iter (int): maximum number of iterations before break
        Returns:
            None (updates label matrix).
        """
        if self.data_source == 'chat':
            tol = 0.001

        # tol = 0.00001
        # propagation_run_Ys = []

        Y_prev = np.zeros((self.n, self.nc), dtype=np.float)
        for i in range(max_iter):
            if np.abs(self.Y-Y_prev).sum() < tol: break
            Y_prev = self.Y
            self.Y = np.matmul(self.T, self.Y) # Y <- TY
            self.Y[:self.nl] = self.Y_static # clamp labels
            # propagation_run_Ys.append(self.Y[self.nl:] / self.Y[self.nl:].sum(axis=1)[:, np.newaxis])
        else:
            warnings.warn(f'max_iter ({max_iter}) was reached without convergence')
        self.Y[self.nl:] /= self.Y[self.nl:].sum(axis=1)[:, np.newaxis] # normalise predictions
        # data_for_plotting.append(propagation_run_Ys)
    
    def recursive(self, x_l, y_l, x_u, y_u, tol=0.0001, max_iter=10000, sigma=None):
        """
        Recursive LP variant (performs propagation, adds labels above
            a certain threshold, and recurs on new labeled dataset).
        Args:
            x_l (list): encoded sentences from labeled data.
            y_l (list(float)): corresponding labels
            x_u (list): encoded sentences from unlabeled data.
            tol (float): convergence condition.
            max_iter (int): maximum # iters before break.
        Returns:
            Classifications & their indices in original unlabeled dataset.
        """
        x_indices = {str(x):i for i, x in enumerate(x_u)}
        xl, yl, xu, yu = x_l, y_l, x_u, y_u
        all_classifications, all_indices = [], []
        last_unlabeled_count = len(x_u)
        iters = 0

        data_for_plotting = []
        recursion_indices = []
        recursion_classifications = []

        while True:
            iters += 1
            self.propagate(tol, max_iter, data_for_plotting=data_for_plotting)
            classifications, indices = self.classify(threshold=True)
            all_classifications += classifications
            for i in indices:
                all_indices.append(x_indices[str(xu[i])])
            
            # recursion_indices.append([x_indices[str(xu[i])] for i in indices])
            # recursion_classifications.append(classifications)

            # create new data for next round.
            for i in indices:
                xl = np.concatenate((xl, np.expand_dims(xu[i],axis=0)), axis=0)
                yl = np.append(yl, [yu[i]])
            xu = np.array([x for i,x in enumerate(xu) if i not in indices])
            yu = np.array([y for i,y in enumerate(yu) if i not in indices])

            if last_unlabeled_count - len(xu) == 0 or len(xu) == 0:
                break
            last_unlabeled_count = len(xu)
            self.__init__(xl, yl, xu, yu, self.nc, sigma=sigma) # last in loop because we have to init before calling recursive
        
        # pickle.dump(data_for_plotting, Path('./paper/propagation_data.pkl').open('wb'))
        # pickle.dump(recursion_indices, Path('./paper/indices_data.pkl').open('wb'))
        # pickle.dump(recursion_classifications, Path('./paper/classifications_data.pkl').open('wb'))
        return all_classifications, all_indices

    def classify(self, threshold=True):
        """
        Extracts classifications from label matrix.
        Args:
            threshold (bool): indicating whether to use threshold variant.
        Returns:
            Predicted classes, and indices of the data labeled with those
                classes (if we use threshold, not all unlabeled data are used).
        """
        Yu = self.Y[self.nl:]
        if threshold:
            indices = np.squeeze(np.argwhere(np.max(Yu, axis=1) >= 0.99), axis=1)
            preds = np.argmax(Yu[indices,:], axis=1)
        else:
            indices = [i for i in range(self.nu)]
            preds = np.argmax(Yu, axis=1)

        return list(preds), list(indices)


def entropy_heuristic(x_l, y_l, x_u, y_u, nc, data_source):
    lp = LabelProp(x_l, y_l, x_u, y_u, nc, data_source=data_source)
    mst_sigma = lp.sigma
    epsilon = 5e-35
    min_sigma, max_sigma = 0.03, mst_sigma
    sigmas, entropies = [], []
    raynge = np.arange(min_sigma, max_sigma, 0.001)
    for sigma in raynge:
        lp = LabelProp(x_l, y_l, x_u, y_u, nc, data_source=data_source, sigma=sigma, epsilon=epsilon)
        lp.propagate()
        Y_u = lp.Y[lp.nl:]
        H = np.sum([entropy(row) for row in Y_u])
        sigmas.append(sigma); entropies.append(H)

    min_entropy = np.argmin(entropies)
    min_entropy_sigma = sigmas[min_entropy]
    lp = LabelProp(x_l, y_l, x_u, y_u, nc, data_source=data_source, sigma=min_entropy_sigma, epsilon=0)
    lp.propagate()
    preds, indices = lp.classify(threshold=False)
    aug_acc = np.sum(preds == y_u[indices]) / len(preds)
    frac_used = float(len(preds)) / len(y_u)

    return mst_sigma, min_entropy, min_entropy_sigma, aug_acc, frac_used
