
import torch
import sys
import time
from torch.utils.cpp_extension import load

#import bit2byte

torch.cuda.set_device(0)
bit2byte = load(name="bit2byte", sources=["bit2byte.cpp"], verbose=True)

cpu = torch.randn(32,1)
gpu = cpu.cuda()

print('cpu',cpu)
print('gpu',gpu)


cpu = cpu.int()
gpu = gpu.int()
cpu_result = bit2byte.packing(cpu)

gpu_result = bit2byte.packing(gpu)

print('cpu_result',cpu_result)
print('gpu_result',gpu_result)