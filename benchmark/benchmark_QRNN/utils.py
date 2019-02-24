import torch
import math
import torch.distributed as dist
import numpy as np

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args, epoch):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def batchify_distributed(data, bsz, args, epoch):
    #shffule the start pointer
    np.random.seed(epoch)
    pointer = np.random.randint(0,len(data))
    data = torch.cat((data[pointer:],data[0:pointer]), dim=0)
    #distributed dataset 
    num_replicas = dist.get_world_size()
    rank = dist.get_rank()
    num_samples = int(math.ceil(data.size(0) * 1.0 / num_replicas))
    total_size = num_samples * num_replicas
    data = torch.cat((data, data[:(total_size - data.size(0))]), dim=0)
    #data += data[:(total_size - data.size(0))]
    assert data.size(0) == total_size

    #divide dataset
    offset = num_samples * rank
    data = data[offset:offset + num_samples]
    assert len(data) == num_samples

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
