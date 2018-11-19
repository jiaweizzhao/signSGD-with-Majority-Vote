import codings
import compressor
import QSGD_cpu
import QSGD_gpu

import torch
import time
import numpy as np

m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
sample = m.sample(torch.Size([1000000]))
sample = sample.cuda()

compressor = compressor.compressor(using_cuda = True)
QSGD_cpu_coder = QSGD_cpu.QSGD()
ATOMO_coder = codings.svd.SVD(random_sample=False, rank=1,
                                   compress=True)

for _ in range(5):
    #QSGD_gpu
    torch.cuda.synchronize()
    QSGD_gpu_start = time.time()
    QSGD_gpu_coding_start = time.time()

    code, data = QSGD_gpu.encode(sample)

    torch.cuda.synchronize()
    QSGD_gpu_coding_time = time.time() - QSGD_gpu_coding_start
    QSGD_gpu_decoding_start = time.time()

    result = QSGD_gpu.decode(code, cuda = True)

    torch.cuda.synchronize()
    QSGD_gpu_decoding_time = time.time() - QSGD_gpu_decoding_start
    QSGD_gpu_time = time.time() - QSGD_gpu_start

    #QSGD_cpu
    torch.cuda.synchronize()
    QSGD_cpu_start = time.time()
    QSGD_cpu_coding_start = time.time()

    code = QSGD_cpu_coder.encode(sample)

    torch.cuda.synchronize()
    QSGD_cpu_coding_time = time.time() - QSGD_cpu_coding_start
    QSGD_cpu_decoding_start = time.time()

    result = QSGD_cpu_coder.decode(code, cuda = True)

    torch.cuda.synchronize()
    QSGD_cpu_decoding_time = time.time() - QSGD_cpu_decoding_start
    QSGD_cpu_time = time.time() - QSGD_cpu_start

    #ATOMO
    torch.cuda.synchronize()
    ATOMO_start = time.time()
    ATOMO_coding_start = time.time()

    sample_np = sample.cpu().numpy().astype(np.float32)
    code = ATOMO_coder.encode(sample_np)

    torch.cuda.synchronize()
    ATOMO_coding_time = time.time() - ATOMO_coding_start
    ATOMO_decoding_start = time.time()

    result = ATOMO_coder.decode(code)

    torch.cuda.synchronize()
    ATOMO_decoding_time = time.time() - ATOMO_decoding_start
    ATOMO_time = time.time() - ATOMO_start

    #Signum
    torch.cuda.synchronize()
    Signum_start = time.time()
    Signum_coding_start = time.time()

    temp, tensor_size = compressor.compress(sample)

    torch.cuda.synchronize()
    Signum_coding_time = time.time() - Signum_coding_start
    Signum_decoding_start = time.time()

    result = compressor.uncompress(temp, tensor_size)

    torch.cuda.synchronize()
    Signum_decoding_time = time.time() - Signum_decoding_start
    Signum_time = time.time() - Signum_start

print('---------')
print('Signum_full_time',Signum_time)
print('Signum_coding_time',Signum_coding_time)
print('Signum_decoding_time',Signum_decoding_time)
print('---------')
print('QSGD_gpu_full_time',QSGD_gpu_time)
print('QSGD_gpu_coding_time',QSGD_gpu_coding_time)
print('QSGD_gpu_decoding_time',QSGD_gpu_decoding_time)
print('---------')
print('QSGD_cpu_full_time',QSGD_cpu_time)
print('QSGD_cpu_coding_time',QSGD_cpu_coding_time)
print('QSGD_cpu_decoding_time',QSGD_cpu_decoding_time)
print('---------')
print('ATOMO_full_time',ATOMO_time)
print('ATOMO_coding_time',ATOMO_coding_time)
print('ATOMO_decoding_time',ATOMO_decoding_time)
print('---------')
