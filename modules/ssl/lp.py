import math
import numpy as np
import warnings

from modules.utilities import euclid


class LabelProp():
    def __init__(self, x_l, y_l, x_u, nc):
        """
        Class for implementation of original label propagation algorithm.
        """
        self.nl, self.nu, self.n = len(x_l), len(x_u), len(x_l)+len(x_u)
        self.nc = nc
        self.T = np.zeros((self.n,self.n), dtype=np.float)
        self.Y = np.zeros((self.n,self.nc), dtype=np.float)

        # initialise T
        ss = math.pow(self._calculate_sigma(x_l,y_l), 2)
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

    def _calculate_sigma(self, x_l, y_l): # MST METHOD
        D = np.zeros((self.nl,self.nl), dtype=np.float)
        for i, u in enumerate(x_l):
            for j, v in enumerate(x_l):
                D[i,j] = euclid(u, v)
        
        min_dist = float('inf')
        for i in range(self.nl):
            for j, dist in enumerate(D[i]):
                if y_l[i] != y_l[j] and dist != 0 and dist < min_dist:
                    min_dist = dist
            
        return min_dist / 3
    
    def _lp_dist(self, u, v, ss):
        return np.exp(-(euclid(u,v)/ss))
    
    def propagate(self, tol=0.0001, max_iter=10000):
        # iterative solution
        Y_prev = np.zeros((self.n,self.nc),dtype=np.float)
        for i in range(max_iter):
            if np.abs(self.Y-Y_prev).sum() < tol: break
            Y_prev = self.Y
            self.Y = np.matmul(self.T, self.Y) # Y <- TY
            # self.Y[self.nl:] /= self.Y[self.nl:].sum(axis=1)[:, np.newaxis] # row normalise Y_u
            self.Y[:self.nl] = self.Y_static # clamp labels
        else:
            warnings.warn(f'max_iter ({max_iter}) was reached without convergence')
        
        self.Y[self.nl:] /= self.Y[self.nl:].sum(axis=1)[:, np.newaxis] # normalise predictions

        # # optimisation solution (inverse can't be calculated here => need to work this out some other way...)
        # T_uu = self.T[self.nl:,self.nl:]
        # T_ul = self.T[self.nl:,:self.nl]
        # Y_l = self.Y[:self.nl]
        # I = np.identity(self.nu, dtype=np.float)
        # print(I.shape)
        # print(T_uu.shape)
        # print((I-T_uu).shape)
        # a = np.linalg.inv((I - T_uu))
        # b = np.matmul(a, T_ul)
        # c = np.matmul(b, Y_l)
        # self.Y[self.nl:] = c # solution for Y_uu
    
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
 
            # print(self.n,self.nl,self.nu)
            # print(f'aug frac: {float(len(classifications)/len(xu))}')
            # aug_acc = 0 if not classifications else np.sum(classifications == yu[indices])/len(classifications)
            # print(f'aug acc: {aug_acc}')

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
        
        print(f'aug frac: {float(len(all_classifications)/len(x_u))}')
        print(f'aug acc: {np.sum(all_classifications == y_u[all_indices])/len(all_classifications)}')
        return all_classifications, all_indices
    
    def p1nn(self, x_l, y_l, x_u):
        classifications = list(-1 * np.ones(self.nu, dtype=np.float))
        x_l_labels = {i:y_l[i] for i,_ in enumerate(x_l)}
        
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
    
    def classify(self, threshold=False):
        # CLASS MASS NORMALIZATION
        # class_proportions_oracle = [0.017, 0.025, 0.017, 0.008, 0.017, 0.017, 0.008, 0.008, 0.116, 0.033, 0.05, 0.174, 0.058, 0.058, 0.017, 0.05, 0.033, 0.058, 0.083, 0.066, 0.091]
        # # class_proportions_train = [0.018, 0.023, 0.01, 0.005, 0.016, 0.009, 0.006, 0.006, 0.123, 0.031, 0.053, 0.183, 0.055, 0.055, 0.012, 0.048, 0.036, 0.061, 0.085, 0.069, 0.093]
        # for i, cp in enumerate(class_proportions_oracle):
        #     self.Y[:,i] *= cp
        # self.Y[self.nl:] /= self.Y[self.nl:].sum(axis=1)[:, np.newaxis] # normalise predictions

        # LABEL BIDDING

        # MAXIMUM LIKELIHOOD (nothing extra before classification)
        
        if threshold:
            indices, preds = [], []
            for i, row in enumerate(self.Y[self.nl:]):
                if np.max(row) == 1: # i.e. if the algorithm is 'certain' (this threshold might need to be different for different encoding/similarity measures)
                    indices.append(i)
                    preds.append(np.argmax(row))
            return preds, indices

        return np.argmax(self.Y[self.nl:], axis=1)


