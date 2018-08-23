#!/usr/bin/env python
import torch
from torch.utils.cpp_extension import load
import time

class compressor:
    def __init__(self, using_cuda = False, local_rank = 0):
        self.bit2byte = load(name="bit2byte", sources=["bit2byte.cpp"], verbose=True)
        self.using_cuda = using_cuda
        self.local_rank = local_rank
        if using_cuda:            
            self.device = torch.device('cuda:' + str(local_rank))
        else:
            self.device = torch.device('cpu')

    def packing(self, src_tensor):
        src_tensor = src_tensor.view(-1)
        src_len = len(src_tensor)
        add_elm = 32 - (src_len % 32)
        if src_len % 32 == 0:
            add_elm = 0
        new_tensor = torch.zeros([add_elm], dtype=torch.float32, device=self.device)
        src_tensor = torch.cat((src_tensor, new_tensor), 0)
        src_tensor = src_tensor.view(32,-1)
        src_tensor = src_tensor.to(dtype=torch.int32)
        dst_tensor = self.bit2byte.packing(src_tensor)

        return dst_tensor, src_tensor.size()

    def unpacking(self, src_tensor, src_tensor_size):
        src_element_num = self.element_num(src_tensor_size)
        add_elm = 32 - (src_element_num % 32)
        if src_element_num % 32 == 0:
            add_elm = 0
        src_tensor = src_tensor.int()
        new_tensor = torch.ones(src_element_num + add_elm, device=self.device, dtype=torch.int32)
        new_tensor = new_tensor.view(32,-1)
        new_tensor = self.bit2byte.unpacking(src_tensor,new_tensor)
        new_tensor = new_tensor.view(-1)
        new_tensor = new_tensor[:src_element_num]
        return new_tensor.view(src_tensor_size)

    def element_num(self, size):
        num = 1
        for i in range(len(size)):
            num *= size[i]
        return num


    def compress(self, src_tensor):
        src_tensor = torch.sign(src_tensor)
        return self.packing(src_tensor)

    def uncompress(self, src_tensor, src_tensor_size):
        dst_tensor = self.unpacking(src_tensor,src_tensor_size)
        dst_tensor = - dst_tensor.add_(-1)
        dst_tensor = dst_tensor.float()
        return dst_tensor











