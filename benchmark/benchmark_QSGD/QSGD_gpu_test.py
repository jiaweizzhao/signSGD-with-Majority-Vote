import torch
import QSGD_gpu
#import QSGD_gpu_level_2 as QSGD_gpu

#a = torch.randn(10)
a = torch.Tensor([1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.10])

print('a',a)
coded = QSGD_gpu.encode(a, True)
print('coded',coded)
b = QSGD_gpu.decode(coded)
print('b',b)



