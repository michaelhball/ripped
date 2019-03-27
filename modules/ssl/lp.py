import matplotlib.pyplot as plt
import warnings

from scipy.stats import entropy

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

    return 1 - (sim/5)


MODEL = None


class LabelProp():
    """
    Class for implementation of label propagation algorithm.
    Args:
        x_l (list(emb)): labeled data in embedded space
        y_l (list(int)): labels for labeled data
        x_u (list(emb)): unlabeled data in embedded space
        sigma (float): parameter controlling edge-weighting
        sim_measure (str): cosine | sts
        source (str): source of STS model for sts sim (sick|stsbenchmark|both)
    """
    def __init__(self, x_l, y_l, x_u, nc, sigma=None, sim_measure='cosine', source=None):
        if sim_measure == "sts":
            global MODEL
            if MODEL is None:
                MODEL = get_sts_model(source)
        self.sim_measure = sim_measure

        self.nl, self.nu, self.n, self.nc = len(x_l), len(x_u), len(x_l)+len(x_u), nc
        self.sigma = sigma if sigma is not None else self._mst_heuristic_sigma(x_l,y_l)
        self._initialise(x_l, y_l, x_u, self.sigma)

    def _initialise(self, x_l, y_l, x_u, sigma, epsilon=0):
        """
        Initialises transition and label matrices.
        Args:
            x_l, y_l (list): encoded sentences and corresponding labels
            x_u (list): encoded sentences from unlabeled data
            sigma (float): Sigma parameter to use in weighting
        Returns:
            None (creates class variables)
        """
        self.ss = math.pow(sigma, 2) # used in weighting function

        # initialise Y
        self.Y = np.zeros((self.n, self.nc), dtype=np.float)
        for i, _ in enumerate(x_l):
            for j in range(self.nc):
                self.Y[i,j] = 1 if j == y_l[i] else 0
        self.Y_static = self.Y[:self.nl]

        # initialise T
        self.T = np.zeros((self.n, self.n), dtype=np.float)
        for i, u in enumerate(x_l):
            for j, v in enumerate(x_l):
                self.T[i,j] = self._weight(u, v)
            for k, z in enumerate(x_u):
                dist = self._weight(u, z)
                self.T[i,self.nl+k] = dist
                self.T[self.nl+k,i] = dist
        for i, u in enumerate(x_u):
            for j, v in enumerate(x_u):
                self.T[self.nl+i,self.nl+j] = self._weight(u, v)
        self.T /= self.T.sum(axis=0)[np.newaxis,:] # column norm
        self.T /= self.T.sum(axis=1)[:,np.newaxis] # row norm

        # epsilon-interpolation smoothing w uniform transition matrix
        U = float(1/self.n) * np.ones((self.n, self.n), dtype=np.float)
        self.T = epsilon * U + (1 - epsilon) * self.T

    def _weight(self, u, v):
        """
        Calculates weight between two nodes in the graph.
        Args:
            u,v (list(float)): two encoded points
        Returns:
            Weight of edge connecting these points.
        """
        return np.exp(-(self._dist_func(u,v)/self.ss))

    def _dist_func(self, u, v):
        """
        The distance function, either based on the inverse of cosine
            or sts similarity.
        Args:
            u,v = list(float): encoded sentences
        Returns:
            Distance between input points.
        """
        if self.sim_measure == "cosine":
            return euclid(u, v)
        elif self.sim_measure == 'sts':
            return sts_dist(MODEL, u, v)

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
            self.__init__(xl, yl, xu, self.nc)
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

            if last_unlabeled_count - len(xu) == 0:
                break
            last_unlabeled_count = len(xu)
        
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
        indices, preds = [], []
        for i, row in enumerate(self.Y[self.nl:]):
            THRESH = .99 if threshold else 0
            if np.max(row) >= THRESH:
                indices.append(i)
                preds.append(np.argmax(row))

        return preds, indices
