#  import torch
import numpy as np
import numpy.linalg as LA
import warnings
import sys
import time
import torch

from .coding import Coding
from .utils import nuclear_indicator, l1_indicator

def _resize_to_2d(x):
    """
    x.shape > 2
    If x.shape = (a, b, *c), assumed that each one of (a, b) pairs has relevant information in c.
    """
    shape = x.shape
    if x.ndim == 1:
        n = x.shape[0]
        return x.reshape((n//2, 2))
    if all([s == 1 for s in shape[2:]]):
        return x.reshape((shape[0], shape[1]))
    # each of (a, b) has related features
    x = x.reshape((shape[0], shape[1], -1))
    # stack those related features into a tall matrix
    x_tmp = x.reshape((shape[0]*shape[1], -1))
    tmp_shape = x_tmp.shape
    return x_tmp.reshape((int(tmp_shape[0]/2), int(tmp_shape[1]*2)))


def _sample_svd(s, rank=0):
    if s[0] < 1e-6:
        return [0], np.array([1.0])
    probs = s / s[0] if rank == 0 else rank * s / s.sum()
    for i, p in enumerate(probs):
        if p > 1:
            probs[i]=1
    sampled_idx = []
    sample_probs = []
    for i, p in enumerate(probs):
        #if np.random.rand() < p:
        # random sampling from bernulli distribution
        if np.random.binomial(1, p):
            sampled_idx += [i]
            sample_probs += [p]
    rank_hat = len(sampled_idx)
    if rank_hat == 0:  # or (rank != 0 and np.abs(rank_hat - rank) >= 3):
        return _sample_svd(s, rank=rank)
    return np.array(sampled_idx, dtype=int), np.array(sample_probs)


class SVD(Coding):
    def __init__(self, compress=True, rank=0, random_sample=True,
                 fetch_indicator=None, *args, **kwargs):
        self.svd_rank = rank
        self.random_sample = random_sample
        self.compress = compress
        # fetch indicator or not
        self.__fetch_indicator = fetch_indicator

    def encode(self, grad, **kwargs):
        # move to CPU; SVD is 5x faster on CPU (at least in torch)
        if not self.compress:
            shape = list(grad.shape)
            return {'grad': grad, 'encode': False}#, {}

        orig_size = list(grad.shape)
        ndims = grad.ndim
        reshaped_flag = False
        if ndims != 2:
            grad = _resize_to_2d(grad)
            shape = list(grad.shape)
            ndims = len(shape)
            reshaped_flag = True

        if ndims == 2:
            u, s, vT = LA.svd(grad, full_matrices=False)

            if self.__fetch_indicator:
                nuclear_indicator = nuclear_indicator(grad, s)
                l1_indicator = l1_indicator(grad)
                print("Step: {}, Nuclear Indicator: {}, L1 Indicator: {}".format(
                    kwargs['step'], nuclear_indicator, l1_indicator))

            if self.random_sample:
                i, probs = _sample_svd(s, rank=self.svd_rank)
                u = u[:, i]
                s = s[i] / probs
                #  v = v[:, i]
                vT = vT[i, :]
            elif self.svd_rank > 0:
                u = u[:, :self.svd_rank]
                s = s[:self.svd_rank]
                #  v = v[:, :self.svd_rank]
                vT = vT[:self.svd_rank, :]

            return {'u': u, 's': s, 'vT': vT, 'orig_size': orig_size,
                    'reshaped': reshaped_flag, 'encode': True,
                    'rank': self.svd_rank}
        return {'grad': grad, 'encode': False}

    def decode(self, encode_output, cuda=False, **kwargs):
        if isinstance(encode_output, tuple) and len(encode_output) == 1:
            encode_output = encode_output[0]
        encode = encode_output.get('encode', False)
        if not encode:
            grad = encode_output['grad']
            grad = torch.Tensor(grad)
            if cuda:
                grad = grad.cuda(async=True)
            return grad

        u, s, vT = (encode_output[key] for key in ['u', 's', 'vT'])
        #grad = u @ np.diag(s) @ vT
        grad = np.dot(np.dot(u, np.diag(s)), vT)
        grad = torch.Tensor(grad)
        grad = grad.view(encode_output['orig_size'])
        if cuda:
            grad = grad.cuda(async=True)
        return grad

        if isinstance(u, np.ndarray):
            u = torch.Tensor(u)
        if isinstance(s, np.ndarray):
            s = torch.Tensor(s)
        if isinstance(vT, np.ndarray):
            vT = torch.Tensor(vT)
        if cuda:
            u = u.contiguous().cuda(async=True)
            s = s.contiguous().cuda(async=True)
            vT = vT.contiguous().cuda(async=True)
            #  u = u.cuda(async=True)
            #  s = s.cuda(async=True)
            #  vT = vT.cuda(async=True)
        #grad_approx = u @ torch.diag(s) @ vT
        grad_approx = np.dot(np.dot(u, torch.diag(s)), vT)
        if encode_output.get('reshaped', False):
            grad_approx = grad_approx.view(encode_output['orig_size'])
        return grad_approx