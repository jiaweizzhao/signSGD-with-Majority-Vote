import numpy as np
from scipy import stats
import torch
import time


def encode(v, **kwargs):
    norm = torch.norm(v)
    w = v.view(-1)

    t = [time.time()]
    signs = torch.sign(w).int()
    probs = torch.abs(w) / norm
    mask = torch.distributions.Bernoulli(probs).sample().byte()
    t += [time.time()]
    idx = torch.arange(0, len(w))
    t += [time.time()]
    if v.is_cuda:
        idx = idx.cuda()
        mask = mask.cuda()
    t += [time.time()]

    selected = torch.masked_select(idx, mask).long()
    signs = torch.masked_select(signs, mask)
    t += [time.time()]

    data = {'masking_time': t[-1] - t[-2], 'gen_mask_time': t[1] - t[0],
            'to_gpu_time': t[-2] - t[-3]}
    return {'signs': signs, 'size': v.size(), 'selected': selected,
            'norm': norm}, data


def decode(code, cuda=False):
    v = torch.zeros(code['size'])
    signs = code['signs']
    if cuda:
        v = v.cuda()
        signs = signs.cuda()
    flat = v.view(-1)
    if len(code['selected']) > 0:
        flat[code['selected']] = code['norm'] * signs.float()
    return v