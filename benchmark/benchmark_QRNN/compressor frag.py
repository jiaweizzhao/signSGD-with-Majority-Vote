#!/usr/bin/env python
import torch
import bit2byte
from torch.utils.cpp_extension import load
import time


class Time_recorder(object):
    def __init__(self):
        self.time = 0
        self.begin = 0

    def reset(self):
        #self.time = 0
        pass

    def set(self):
        #self.begin = time.time()
        pass

    def record(self):
        #self.end = time.time()
        #self.time += self.end - self.begin
        pass

    def get_time(self):
        return self.time


class compressor(torch.nn.Module):
    def __init__(self, using_cuda = False, local_rank = 0):
        #bit2byte = load(name="bit2byte", sources=["bit2byte.cpp"], verbose=True)
        super(compressor, self).__init__()
        self.para_temp = torch.nn.Parameter(torch.zeros(1))


        self.using_cuda = using_cuda
        local_rank = torch.cuda.current_device()
        self.local_rank = local_rank
        if using_cuda:            
            self.device = torch.device('cuda:' + str(local_rank))
        else:
            self.device = torch.device('cpu')

        self.source_device = torch.device('cuda:0')

        self.compression_python = Time_recorder()
        self.compression_cuda = Time_recorder()
        self.decompression_python = Time_recorder()
        self.decompression_cuda = Time_recorder()
        self.majority_vote_compression = Time_recorder()
        self.majority_vote_decompression = Time_recorder()
        self.majority_vote_sum_calculation = Time_recorder()
        self.compression_specific = Time_recorder()

    def get_time_result(self):
        print('in compressor module:')
        print('compression_python',self.compression_python.get_time())
        print('compression_cuda',self.compression_cuda.get_time())
        print('decompression_python',self.decompression_python.get_time())
        print('decompression_cuda',self.decompression_cuda.get_time())
        print('majority_vote_compression',self.majority_vote_compression.get_time())
        print('majority_vote_decompression',self.majority_vote_decompression.get_time())
        print('majority_vote_sum_calculation',self.majority_vote_sum_calculation.get_time())
        print('compression_specific',self.compression_specific.get_time())


    def reset_time(self):
        self.compression_python.reset()
        self.compression_cuda.reset()
        self.decompression_python.reset()
        self.decompression_cuda.reset()
        self.majority_vote_compression.reset()
        self.majority_vote_decompression.reset()
        self.majority_vote_sum_calculation.reset()
        self.compression_specific.reset()


    def packing(self, src_tensor):
        #src_tensor = src_tensor.to(torch.cuda.current_device())
        self.compression_python.set()
        self.compression_specific.set()
        #torch.cuda.synchronize()
        self.compression_specific.record()
        src_tensor = torch.sign(src_tensor)
        src_tensor_size = src_tensor.size()
        src_tensor = src_tensor.view(-1)
        src_len = len(src_tensor)
        add_elm = 32 - (src_len % 32)
        if src_len % 32 == 0:
            add_elm = 0
        new_tensor = torch.zeros([add_elm], dtype=torch.float32, device=torch.cuda.current_device())
        src_tensor = torch.cat((src_tensor, new_tensor), 0)
        src_tensor = src_tensor.view(32,-1)
        src_tensor = src_tensor.to(dtype=torch.int32)
        #torch.cuda.synchronize()
        self.compression_python.record()
        self.compression_cuda.set()
        #torch.cuda.synchronize()
        dst_tensor = bit2byte.packing(src_tensor)
        dst_tensor = dst_tensor.to(dtype=torch.int32)
        #torch.cuda.synchronize()
        self.compression_cuda.record()

        return dst_tensor, src_tensor_size

    def unpacking(self, src_tensor, src_tensor_size):
        #src_tensor = src_tensor.to(torch.cuda.current_device())
        self.decompression_python.set()
        #torch.cuda.synchronize()
        src_element_num = self.element_num(src_tensor_size)
        add_elm = 32 - (src_element_num % 32)
        if src_element_num % 32 == 0:
            add_elm = 0
        src_tensor = src_tensor.int()
        new_tensor = torch.ones(src_element_num + add_elm, device=torch.cuda.current_device(), dtype=torch.int32)
        new_tensor = new_tensor.view(32,-1)
        #torch.cuda.synchronize()
        self.decompression_python.record()
        self.decompression_cuda.set()
        #torch.cuda.synchronize()
        new_tensor = bit2byte.unpacking(src_tensor,new_tensor)
        #torch.cuda.synchronize()
        self.decompression_cuda.record()
        self.decompression_python.set()
        #torch.cuda.synchronize()
        new_tensor = new_tensor.view(-1)
        new_tensor = new_tensor[:src_element_num]
        new_tensor = new_tensor.view(src_tensor_size)
        new_tensor = - new_tensor.add_(-1)
        new_tensor = new_tensor.float()
        #torch.cuda.synchronize()

        self.decompression_python.record()

        return new_tensor

    def forward(self, src_tensor_list):
        self.majority_vote_decompression.set()
        src_tensor = src_tensor_list
        #print('part_size',src_tensor.size(),src_tensor)
        #torch.cuda.synchronize()
        voter_num = len(src_tensor_list)
        #src_tensor = torch.stack(src_tensor_list)
        #src_tensor = src_tensor.to(torch.cuda.current_device())
        src_tensor = src_tensor.view(-1)
        full_size = 32 * len(src_tensor)
        new_tensor = torch.ones(full_size, device=torch.cuda.current_device(), dtype=torch.int32)
        new_tensor = new_tensor.view(32,-1)
        #print(src_tensor)
        #print(new_tensor)
        new_tensor = bit2byte.unpacking(src_tensor,new_tensor)
        new_tensor = - new_tensor.add_(-1)
        #torch.cuda.synchronize()
        self.majority_vote_decompression.record()

        #sum
        self.majority_vote_sum_calculation.set()
        #torch.cuda.synchronize()
        new_tensor = new_tensor.permute(1,0).contiguous().view(voter_num,-1)
        new_tensor = torch.sum(new_tensor,0)
        new_tensor = new_tensor.view(-1,32).permute(1,0)
        #torch.cuda.synchronize()
        self.majority_vote_sum_calculation.record()

        self.majority_vote_compression.set()
        #torch.cuda.synchronize()
        new_tensor = torch.sign(new_tensor)
        new_tensor = bit2byte.packing(new_tensor)
        new_tensor = new_tensor.to(dtype=torch.int32)
        #torch.cuda.synchronize()
        self.majority_vote_compression.record()

        new_tensor = new_tensor.expand(1,-1)
        return new_tensor


    def element_num(self, size):
        num = 1
        for i in range(len(size)):
            num *= size[i]
        return num


    def compress(self, src_tensor):
        #src_tensor = torch.sign(src_tensor)
        return self.packing(src_tensor)

    def uncompress(self, src_tensor, src_tensor_size):
        dst_tensor = self.unpacking(src_tensor,src_tensor_size)
        #dst_tensor = - dst_tensor.add_(-1)
        #dst_tensor = dst_tensor.float()
        return dst_tensor











