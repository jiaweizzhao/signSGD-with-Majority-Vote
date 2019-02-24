import numpy as np
import numpy.linalg as LA
from scipy import stats
import torch
import time
from coding import Coding


class QSGD(Coding):

    def __init__(self, *args, scheme='qsgd', **kwargs):
        self.scheme = scheme
        super().__init__(*args, **kwargs)

    def encode(self, v, **kwargs):
        if isinstance(v, (torch.Tensor, torch.cuda.FloatTensor)):
            w = v.cpu().numpy().flat[:]
        elif isinstance(v, np.ndarray):
            w = v.flat[:]
        else:
            raise ValueError("Object passed to encode not ndarray or torch.Tensor")

        if self.scheme == 'qsgd':
            norm = LA.norm(v)
        elif self.scheme == 'terngrad':
            norm = np.linalg.norm(w, ord=np.inf)
            limit = grad_clip_limit(w, clip_factor=2.5)
            w = np.clip(w, -limit, limit)

        signs = np.sign(w).astype('int')
        probs = np.abs(w) / norm
        #  mask = stats.bernoulli.rvs(probs).astype('bool')  # rvs is not thread-safe
        mask = np.random.rand(len(probs)) < probs
        # generate 0/1 random variable with mean prob
        # or, 0 if x < prob and 1 if x >= prob
        idx = np.arange(len(w), dtype='uint32')

        selected = idx[mask].astype('uint32')
        signs = signs[mask].astype('int8')
        signs = ((signs + 1) / 2).astype('bool')

        code = {'signs': signs, 'shape': v.shape, 'selected': selected,
                'norm': norm}

        if kwargs.pop('timings', False):
            data = {}
            return code, data
        return code

    def decode(self, code, cuda=False, implementation='numpy', codes=[], **kwargs):
        """
        Decode the coding.
        ## NumPy
         'comm_wait': 0.0728750228881836,
         'decode_time': 0.1349341869354248,
         'example_to_gpu': 0.0006515979766845703,
         'grad_compute_time': 0.5815503597259521,
         'grad_forward_pass': 0.23496603965759277,
         'grad_variance_increase': 31.754316389320049,
         'iallgather_prepare_time': 0.017401456832885742,
         'isend_time': 0.029105424880981445,
        ## PT GPU
        """
        if self.scheme == 'terngrad' and len(codes) > 0:
            code['norm'] = self._get_max_norm(codes)

        if implementation == 'numpy':
            v = np.zeros(code['shape'])
            signs = np.array(code['signs'], dtype='int8')
            signs = signs*2 - 1
            selected = np.array(code['selected'], dtype='int16')
            #  selected = torch.LongTensor(selected)

            if len(selected) > 0:
                v.flat[selected] = code['norm'] * signs
        else:
            raise ValueError('Whoops, implementation')
        v = torch.Tensor(v)
        if cuda:
            v = v.cuda()
        return v

    def _get_max_norm(self, codes):
        scalars = [code['norm'] for code in codes]
        return max(scalars)


def grad_clip_limit(grad, clip_factor=2.5):
    """ Get the scalers."""
    if clip_factor > 1.0e-5:
        return clip_factor * np.std(grad.flat[:])
    return np.max(np.abs(grad.flat[:]))