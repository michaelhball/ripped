"""
Math utilities.
"""
import math
import numpy as np
import scipy

__all__ = ['cosine', 'confidence_interval', 'euclid', 'softmax']


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def euclid(u,v):
    return np.linalg.norm(u-v)


def softmax(u):
    return scipy.special.softmax(u)


def confidence_interval(p, std, n):
    return scipy.stats.t.ppf(p, n) * (std / math.sqrt(n))
