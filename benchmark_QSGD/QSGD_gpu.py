import numpy as np
from scipy import stats
import torch
import time

#enable_max = False
#level = 5

def encode(v, enable_max = False, level = 1, **kwargs):
    if enable_max:
        norm = torch.max(torch.abs(v))
    else:
        norm = torch.norm(v)

    w = v.view(-1)

    #print('the size of w', w.size())
    idx = torch.arange(0, len(w)).type_as(w)
    signs = torch.sign(w).type_as(w)
    full_index = torch.Tensor([]).long()
    full_value = torch.Tensor([]).float()
    if w.is_cuda:
        full_index = full_index.cuda()
        full_value = full_value.cuda()


    for interval_l in range(level):  #range should be [interval_l/level, (interval_l + 1)/level]

        lower_bound = interval_l/level * norm
        upper_bound = (interval_l + 1)/level * norm
        interval_mask = w.clone().byte()
        if w.is_cuda:
            interval_mask = interval_mask.cuda()
        interval_mask[:] = 1
        interval_mask[torch.abs(w) <= lower_bound] = 0
        interval_mask[torch.abs(w) >= upper_bound] = 0

        #select interval_l/level * norm
        probs = 1 - (level * torch.abs(w) / norm - interval_l)
        probs[torch.abs(w) <= lower_bound] = 0
        probs[torch.abs(w) >= upper_bound] = 0        
        mask = torch.distributions.Bernoulli(probs).sample().byte()
        if w.is_cuda:
            mask = mask.cuda()
        selected_index = torch.masked_select(idx, mask).long()
        selected_value = torch.masked_select(signs, mask).float() * interval_l/level
        if interval_l/level != 0:
            full_index = torch.cat((full_index, selected_index), 0)
            full_value = torch.cat((full_value, selected_value), 0)

        #(interval_l + 1)/level * norm
        interval_mask[selected_index] = 0
        selected_index = torch.masked_select(idx, interval_mask).long()
        selected_value = torch.masked_select(signs, interval_mask).float() * (interval_l + 1)/level
        full_index = torch.cat((full_index, selected_index), 0)
        full_value = torch.cat((full_value, selected_value), 0)           

    data = False #none data

    return {'size': v.size(), 'signs': full_value, 'selected': full_index, \
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