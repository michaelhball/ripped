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
    def __init__(self, x_l, y_l, x_u, y_u, nc, data_source='chatbot', display=False):
        st = time.time()
        self.data_source = data_source
        self.nl, self.nu, self.n, self.nc = len(x_l), len(x_u), len(x_l)+len(x_u), nc

        # calculate sigma
        sigma, T_ll = self._mst_heuristic_sigma(x_l, y_l)
        sigma = 0.12
        self.ss = math.pow(sigma, 2)
        if display: print('1) computed sigma')

        # initialise T
        self.T = np.zeros((self.n, self.n), dtype=np.float)
        self.T[:self.nl,:self.nl] = T_ll
        if display: print('2.1) computed T_ll')
        T_lu = spatial.distance.cdist(x_l, x_u)
        self.T[:self.nl,self.nl:] = T_lu
        if display: print('2.2) computed T_lu')
        T_ul = np.transpose(T_lu)
        self.T[self.nl:,:self.nl] = T_ul
        if display: print('2.3) computed T_ul')
        T_uu = spatial.distance.cdist(x_u, x_u)
        self.T[self.nl:,self.nl:] = T_uu
        if display: print('2.4) computed T_uu')
        self.T = np.exp(-self.T/self.ss) # weighting function
        self.T /= self.T.sum(axis=0)[np.newaxis,:] # column norm
        self.T /= self.T.sum(axis=1)[:,np.newaxis] # row norm
        if display: print('3) initialised T')

        # # epsilon-interpolation smoothing w uniform transition matrix (MAYBE I'M NOT DOING THIS CORRECTLY SINCE WE CAN'T GET V GOOD RESULTS USING SIGMA-SETTING THING)
        # U = float(1/self.n) * np.ones((self.n, self.n), dtype=np.float)
        # self.T = epsilon * U + (1 - epsilon) * self.T

        # initialise Y
        self.Y = np.zeros((self.n, self.nc), dtype=np.float)
        for i, _ in enumerate(x_l):
            for j in range(self.nc):
                self.Y[i,j] = 1 if j == y_l[i] else 0
        self.Y_static = self.Y[:self.nl]
        if display: print('4) initialised Y')
        if display: print(f'5) completed initialisation in {time.time()-st} seconds')

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
    
    def sigma_fit(self, x_l, y_l, x_u, y_u):
        """
        Function to find optimal sigma for given data
            using entropy of probabilistic output classifications.
        Args:
            x_l (list): encoded sentences from labeled data
            y_l (list): corresponding labels for x_l
            x_u (list): encoded sentences from unlabeled data
        Returns:
            Optimal sigma value for use with given input data (according to lowest entropy heuristic).
        """
        print(f"MST-det sigma: {self.sigma}")
        epsilons = [1e-100, 1e-150]
        colors = ['b', 'g']
        for epsilon, color in zip(epsilons, colors):
            sigmas, entropies = [], []
            min_sigma, max_sigma = 0.001, self.sigma+0.1
            raynge = np.arange(min_sigma, max_sigma, 0.001)
            for sigma in tqdm(raynge, total=len(raynge)):
                self._initialise(x_l, y_l, x_u, sigma=sigma, epsilon=epsilon) # creates T & Y matrices
                self.propagate() # perform propagation
                Y_u = self.Y[self.nl:] # get unlabeled class probs
                H = np.sum([entropy(row) for row in Y_u]) # calc entropy
                sigmas.append(sigma); entropies.append(H)
            
            print(f'epsilon={epsilon}, minimum entropy sigma: {sigmas[np.argmin(entropies)]}')

            self._initialise(x_l, y_l, x_u, sigma=sigmas[np.argmin(entropies)], epsilon=0)
            self.propagate()
            preds, _ = self.classify(threshold=False)
            print(f'aug accuracy: {np.sum(preds == y_u) / len(preds)}\n')

            plt.plot(sigmas, entropies, color, label=f'epsilon={epsilon}')
        
        plt.title("sigma v entropy")
        plt.xlabel('sigma'); plt.ylabel('entropy')
        plt.grid(b=True)
        plt.legend()
        plt.show()

    def propagate(self, tol=0.0001, max_iter=10000):
        """
        Performs label propagation until convergence
            (absolute difference in label matrix).
        Args:
            tol (float): tolerance defining convergence point
            max_iter (int): maximum number of iterations before break
        Returns:
            None (updates label matrix).
        """
        if self.data_source == 'trec':
            tol = 0.1
        elif self.data_source == 'chat':
            tol = 0.001
            
        Y_prev = np.zeros((self.n, self.nc), dtype=np.float)
        for i in range(max_iter):
            if np.abs(self.Y-Y_prev).sum() < tol: break
            Y_prev = self.Y
            self.Y = np.matmul(self.T, self.Y) # Y <- TY
            self.Y[:self.nl] = self.Y_static # clamp labels
        else:
            warnings.warn(f'max_iter ({max_iter}) was reached without convergence')
        self.Y[self.nl:] /= self.Y[self.nl:].sum(axis=1)[:, np.newaxis] # normalise predictions
    
    def recursive(self, x_l, y_l, x_u, y_u, tol=0.0001, max_iter=10000):
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

        while True:
            iters += 1
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

            if last_unlabeled_count - len(xu) == 0 or len(xu) == 0:
                break
            last_unlabeled_count = len(xu)
            self.__init__(xl, yl, xu, yu, self.nc) # last in loop because we have to init before calling recursive
        
        return all_classifications, all_indices
    
    def p1nn(self, x_l, y_l, x_u):
        """
        Performs propagate-1-nearest-neighbour LP variant.
            (at each time step, labels the unlabeled example
            closest to any labeled example - repeats w. new
            dataset).
        Args:
            x_l (list): encoded sentences from labeled data
            y_l (list(float)): corresponding labels
            x_u (list): encoded sentences from unlabeled data
        Returns:
            classification predictions.
        """
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
