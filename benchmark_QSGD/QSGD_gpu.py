import numpy as np
from scipy import stats
import torch
import time

#enable_max = False
level = 2

def encode(v, enable_max, **kwargs):
    if enable_max:
        norm = torch.max(torch.abs(v))
    else:
        norm = torch.norm(v)

    w = v.view(-1)

    #print('the size of w', w.size())


    idx = torch.arange(0, len(w))
    if v.is_cuda:
        idx = idx.cuda()
    #for g > 1/2
    probs = (2 * torch.abs(w) / norm) - 1
    probs[torch.abs(w) < (norm / 2)] = 0
    mask_upper = torch.distributions.Bernoulli(probs).sample().byte()
    if v.is_cuda:
        mask_upper.cuda()
    selected_upper = torch.masked_select(idx, mask_upper).long()
    signs_upper = torch.Tensor(w.size()).float() #change int to float
    signs_upper[w > norm / 2] = 1
    signs_upper[w < - norm / 2] = -1
    signs_upper = torch.masked_select(signs_upper, mask_upper)

    #for g < 1/2
    probs = 2 * torch.abs(w) / norm
    probs[torch.abs(w) > (norm / 2)] = 0
    mask_lower = torch.distributions.Bernoulli(probs).sample().byte()
    if v.is_cuda:
        mask_lower.cuda()
    selected_lower = torch.masked_select(idx, mask_lower).long()
    signs_lower = torch.Tensor(w.size()).float() #change int to float
    signs_lower[torch.abs(w) < norm / 2] = 0.5
    signs_lower[w < 0] = - signs_lower[w < 0]
    signs_lower = torch.masked_select(signs_lower, mask_lower)

    return {'size': v.size(), 'signs_upper': signs_upper, 'selected_upper': selected_upper, \
        'signs_lower': signs_lower, 'selected_lower': selected_lower, 'norm': norm}

def decode(code, cuda=False):
    v = torch.zeros(code['size'])
    signs_upper = code['signs_upper']
    signs_lower = code['signs_lower']
    if cuda:
        v = v.cuda()
        signs_upper = signs_upper.cuda()
        signs_lower = signs_lower.cuda()
    flat = v.view(-1)
    if len(code['selected_upper']) > 0:
        flat[code['selected_upper']] = code['norm'] * signs_upper.float()
    if len(code['selected_lower']) > 0:
        flat[code['selected_lower']] = code['norm'] * signs_lower.float()

    return v