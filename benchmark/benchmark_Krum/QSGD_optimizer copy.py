#coding=utf-8
#Modified Signum version for ATOMO
#Note: this version only for GPU tensor
import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
import compressor
import time
import os

import QSGD_gpu
import numpy as np


#add time recorder
class Time_recorder(object):
    def __init__(self):
        self.time = 0
        self.begin = 0

    def reset(self):
        self.time = 0
        pass

    def set(self):
        torch.cuda.synchronize()
        self.begin = time.time()
        #pass

    def record(self):
        torch.cuda.synchronize()
        self.end = time.time()
        self.time += self.end - self.begin
        #pass

    def get_time(self):
        return self.time

a = Time_recorder()
b = Time_recorder()
c = Time_recorder()
d = Time_recorder()
e = Time_recorder()
f = Time_recorder()



class SGD_distribute(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay = 0, compression_buffer = False, all_reduce = False ,local_rank = 0, gpus_per_machine = 1, args = None, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay)

        super(SGD_distribute, self).__init__(params, defaults)

        #custom code
        self.compression_buffer = compression_buffer
        self.all_reduce = all_reduce
        self.device = torch.device('cuda:' + str(local_rank))

        self.MB = 1024 * 1024
        self.bucket_size = 100 * self.MB

        #parameter for ATOMO
        self.svd_rank = 1
        self.bidirection_compress = args.bidirection_compress
        self.all_gather_commu = args.all_gather_commu

        if self.compression_buffer:

            #self.compressor = codings.svd.SVD(compress=False)
            #self.compressor = codings.lossless_compress.LosslessCompress()
            #self.compressor = codings.svd.SVD(random_sample=False, rank=self.svd_rank,
                                  # compress=True)

            self.local_rank = local_rank
            self.global_rank = dist.get_rank()
            self.local_dst_in_global = self.global_rank - self.local_rank

            self.inter_node_group = []
            self.nodes = dist.get_world_size() // gpus_per_machine

            self.intra_node_group_list = []
            for index in range(self.nodes):
                # set inter_node_group
                self.inter_node_group.append(0 + index * gpus_per_machine)
                # set all intra_node_group
                intra_node_group_temp = []
                for intra_index in range(gpus_per_machine):
                    intra_node_group_temp.append(intra_index + index * gpus_per_machine)
                intra_node_group_temp = dist.new_group(intra_node_group_temp)
                self.intra_node_group_list.append(intra_node_group_temp)

                if self.local_dst_in_global == 0 + index * gpus_per_machine:
                    self.nodes_rank = index


            #self.intra_node_list = self.intra_node_group
            self.inter_node_list = self.inter_node_group
            self.inter_node_group_list = []
            for index in range(len(self.inter_node_list)):
                if index is not 0:
                    temp = dist.new_group([self.inter_node_list[0],self.inter_node_list[index]])
                    self.inter_node_group_list.append(temp)
            self.all_gpu = dist.new_group()

            self.all_inter_node_group = dist.new_group(self.inter_node_list)

            if dist.get_rank() == 0 or dist.get_rank() == 8:
                print('nodes', self.nodes)
                print('intra_node_group_list',self.intra_node_group_list)
                print('inter_node_group',self.inter_node_group_list)
                print('all_inter_node_group', self.inter_node_list)


            #NOTE!!! only for test
            #self.nodes = 1

    def __setstate__(self, state):
        super(SGD_distribute, self).__setstate__(state)

    def pack_len_tensor_into_tensor(self, tensor):
        #tensor here should be 1-dimension
        tensor_len = len(tensor)
        return torch.Tensor([tensor_len], device = self.device)


    def step(self, closure=None):



        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            all_grads = []

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if self.compression_buffer==False:
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    # signum
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    else:
                        buf = param_state['momentum_buffer']

                    buf.mul_(momentum).add_((1 - momentum),d_p)
                    d_p.copy_(buf)
                all_grads.append(d_p)

            dev_grads_buckets = _take_tensors(all_grads, self.bucket_size)
            for dev_grads in dev_grads_buckets:
                d_p_new = _flatten_dense_tensors(dev_grads)

                if self.all_reduce:
                    dist.all_reduce(d_p_new, group = 0) #self.all_gpu
                else:
                    if self.nodes > 1:
                        if self.compression_buffer:
                            coded, data_time = QSGD_gpu.encode(d_p_new)
                            #specific coded dic just on CPU
                            tensor_signs = coded['signs']
                            tensor_selected = coded['selected']
                            tensor_norm = coded['norm']
                            #size
                            tensor_signs_size = self.pack_len_tensor_into_tensor(tensor_signs)
                            tensor_selected_size = self.pack_len_tensor_into_tensor(tensor_selected)
                            #tensor_norm_size = self.pack_len_tensor_into_tensor(tensor_norm) norm doesn't need size

                            #custom
                            '''
                            print(tensor_signs.type())
                            print(tensor_selected.type())
                            print(tensor_norm.type())
                            '''

                        else:
                            d_p_new = torch.sign(d_p_new)

                        if self.local_rank == 0:

                            if self.all_gather_commu:
                                #This version only for instances each with one GPU
                                for node_index in self.inter_node_list:
                                    if node_index != self.nodes_rank:

                                        d.set()
                                        
                                        coded_temp = QSGD_gpu.coded_template()
                                        #coded_temp = coded.copy()
                                        
                                        b.set()
                                        tensor_signs_size_temp = tensor_signs_size.clone()
                                        dist.broadcast(tensor_signs_size_temp, node_index, group = self.all_inter_node_group)
                                        b.record()
                                        c.set()
                                        tensor_signs_temp = torch.zeros([int(tensor_signs_size_temp[0])], device = self.device, dtype=torch.int)
                                        print('tensor_signs_temp', tensor_signs_temp.size())
                                        c.record()
                                        a.set()
                                        dist.broadcast(tensor_signs_temp, node_index, group = self.all_inter_node_group)
                                        a.record()
                                        d.record()

                                        e.set()
                                        tensor_selected_size_temp = tensor_selected_size.clone()
                                        dist.broadcast(tensor_selected_size_temp, node_index, group = self.all_inter_node_group)
                                        tensor_selected_temp = torch.zeros([int(tensor_selected_size_temp[0])], device = self.device, dtype=torch.long)                             
                                        dist.broadcast(tensor_selected_temp, node_index, group = self.all_inter_node_group)
                                        print('tensor_selected_temp', tensor_selected_temp.size())
                                        e.record()
                                        f.set()
                                        tensor_norm_temp = tensor_norm.clone()
                                        dist.broadcast(tensor_norm_temp, node_index, group = self.all_inter_node_group)

                                        coded_temp['signs'] = tensor_signs_temp
                                        coded_temp['selected'] = tensor_selected_temp
                                        coded_temp['norm'] = tensor_norm_temp
                                        coded_temp['size'] = coded['size']

                                        tensor_decoded = QSGD_gpu.decode(coded_temp, cuda = True)
                                        d_p_new = d_p_new + tensor_decoded
                                        f.record()

                                        
                                        print('a', a.get_time())
                                        print('b', b.get_time())
                                        print('c', c.get_time())
                                        print('d', d.get_time())
                                        print('e', e.get_time())
                                        print('f', f.get_time())
                                        

                                    else:
                                        dist.broadcast(tensor_signs_size, node_index, group = self.all_inter_node_group)
                                        dist.broadcast(tensor_signs, node_index, group = self.all_inter_node_group) 
                                        dist.broadcast(tensor_selected_size, node_index, group = self.all_inter_node_group)
                                        dist.broadcast(tensor_selected, node_index, group = self.all_inter_node_group) 
                                        dist.broadcast(tensor_norm, node_index, group = self.all_inter_node_group) 
                                d_p_new = d_p_new / dist.get_world_size()





                            else:
                                if dist.get_rank() == 0:
                                    for index, inter_node_group in enumerate(self.inter_node_group_list):
                                        coded_temp = QSGD_gpu.coded_template()

                                        tensor_signs_size_temp = tensor_signs_size.clone()
                                        dist.broadcast(tensor_signs_size_temp, self.inter_node_list[index + 1], group = inter_node_group)
                                        tensor_signs_temp = torch.randn([int(tensor_signs_size_temp[0])]).type_as(tensor_signs)
                                        dist.broadcast(tensor_signs_temp, self.inter_node_list[index + 1], group = inter_node_group)

                                        tensor_selected_size_temp = tensor_selected_size.clone()
                                        dist.broadcast(tensor_selected_size_temp, self.inter_node_list[index + 1], group = inter_node_group)
                                        tensor_selected_temp = torch.randn([int(tensor_selected_size_temp[0])]).type_as(tensor_selected)                             
                                        dist.broadcast(tensor_selected_temp, self.inter_node_list[index + 1], group = inter_node_group)

                                        tensor_norm_temp = tensor_norm.clone()
                                        dist.broadcast(tensor_norm_temp, self.inter_node_list[index + 1], group = inter_node_group)

                                        coded_temp['signs'] = tensor_signs_temp
                                        coded_temp['selected'] = tensor_selected_temp
                                        coded_temp['norm'] = tensor_norm_temp
                                        coded_temp['size'] = coded['size']

                                        tensor_decoded = QSGD_gpu.decode(coded_temp, cuda = True)
                                        d_p_new = d_p_new + tensor_decoded


                                        '''
                                        #temp
                                        print(tensor_decoded)
                                        tensor_decoded_temp = tensor_decoded.clone()
                                        dist.broadcast(tensor_decoded_temp, self.inter_node_list[index + 1], group = inter_node_group)
                                        if tensor_decoded == tensor_decoded_temp:
                                            print('success')
                                        print(tensor_signs_size_temp)
                                        print(tensor_selected_size_temp)
                                        '''

                                    d_p_new = d_p_new / dist.get_world_size()

                                else:
                                    dist.broadcast(tensor_signs_size, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1])
                                    dist.broadcast(tensor_signs, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1]) 
                                    dist.broadcast(tensor_selected_size, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1])
                                    dist.broadcast(tensor_selected, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1]) 
                                    dist.broadcast(tensor_norm, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1]) 

                                    '''
                                    #temp
                                    tensor_decoded = QSGD_gpu.decode(coded, cuda = True)
                                    print(tensor_decoded)
                                    dist.broadcast(tensor_decoded, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1]) 
                                    print(tensor_signs_size)
                                    print(tensor_selected_size)
                                    '''

                                    dist.barrier(group = self.all_inter_node_group)

                                #os._exit()



                                if self.bidirection_compress:                
                                    if dist.get_rank() == 0:

                                        coded, data_time = QSGD_gpu.encode(d_p_new)
                                        tensor_signs = coded['signs']
                                        tensor_selected = coded['selected']
                                        tensor_norm = coded['norm']

                                        tensor_signs_size = self.pack_len_tensor_into_tensor(tensor_signs)
                                        tensor_selected_size = self.pack_len_tensor_into_tensor(tensor_selected)

                                        dist.barrier(group = self.all_inter_node_group)

                                    dist.broadcast(tensor_signs_size, 0, group = self.all_inter_node_group)
                                    dist.broadcast(tensor_selected_size, 0, group = self.all_inter_node_group)
                                    if dist.get_rank() != 0:
                                        tensor_signs = torch.randn([int(tensor_signs_size[0])]).type_as(tensor_signs)
                                        tensor_selected = torch.randn([int(tensor_selected_size[0])]).type_as(tensor_selected)

                                    dist.barrier(group = self.all_inter_node_group)

                                    dist.broadcast(tensor_signs, 0, group = self.all_inter_node_group)
                                    dist.broadcast(tensor_selected, 0, group = self.all_inter_node_group)
                                    dist.broadcast(tensor_norm, 0, group = self.all_inter_node_group)

                                    coded['signs'] = tensor_signs
                                    coded['selected'] = tensor_selected
                                    coded['norm'] = tensor_norm

                                    tensor_decoded = QSGD_gpu.decode(coded, cuda = True)
                                    d_p_new = tensor_decoded

                                else:
                                    if dist.get_rank() == 0:
                                        dist.barrier(group = self.all_inter_node_group)
                                    dist.broadcast(d_p_new, 0, group = self.all_inter_node_group)

                    else:
                        # test for one
                        coded, data_time = QSGD_gpu.encode(d_p_new)
                        tensor_decoded = QSGD_gpu.decode(coded, cuda = True)
                        d_p_new = tensor_decoded


                #unflatten
                dev_grads_new = _unflatten_dense_tensors(d_p_new,dev_grads)
                for grad, reduced in zip(dev_grads, dev_grads_new):
                    grad.copy_(reduced)
            for p in group['params']:
                if self.compression_buffer:
                    if weight_decay != 0:
                        p.grad.data.add_(weight_decay, p.data)
                p.data.add_(-group['lr'], p.grad.data)

        return loss


