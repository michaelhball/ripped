import numpy as np
import torch

from torch.autograd import Variable


def randomise(arr):
    """
    Randomises the order of the given array along its first dimension
    """
    np.random.seed(42)
    return arr[np.random.permutation(len(arr))]


def map_over(x, f):
    return [f(o) for o in x] if isinstance(x, (list, tuple)) else f(x)


def create_variable(x, volatile, requires_grad=False):
    """
    Creates a Pytorch tensor.
    """
    if type(x) != Variable:
        x = Variable(T(x), requires_grad=requires_grad)
    return x


def V(x, requires_grad=False, volatile=False):
    """
    Creates a single Tensor or list of Tensors depending on input x.
    """
    return map_over(x, lambda o: create_variable(o, requires_grad, volatile))


def T(a, half=False, cuda=False):
    """
    Convert np array into pytorch tensor
    """
    if not torch.is_tensor(a):
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            a = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            a = torch.cuda.HalfTensor(a) if half else torch.FloatTensor(a)
        else:
            raise NotImplementedError(a.dtype)

    return a