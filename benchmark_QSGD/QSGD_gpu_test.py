import torch
import QSGD_gpu

a = torch.randn(100)
print('a',a)
coded = QSGD_gpu.encode(a, True)
print('coded',coded)
b = QSGD_gpu.decode(coded)
print('b',b)

