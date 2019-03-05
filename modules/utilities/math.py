"""
Math utilities.
"""
import math
import numpy as np

from scipy import stats


__all__ = ['cosine', 'confidence_interval']


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def confidence_interval(confidence, std, n):
    t_value = stats.t.ppf(confidence, n)
    return t_value * (std / math.sqrt(n))
