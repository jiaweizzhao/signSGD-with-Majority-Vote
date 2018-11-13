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

import codings
import numpy as np


class SGD_distribute(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay = 0, compression_buffer = False, all_reduce = False ,local_rank = 0, gpus_per_machine = 1, **kwargs):
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

        self.MB = 1024 * 1024
        self.bucket_size = 100 * self.MB

        #parameter for ATOMO
        self.svd_rank = 1

        if self.compression_buffer:

            #self.compressor = codings.svd.SVD(compress=False)
            #self.compressor = codings.lossless_compress.LosslessCompress()
            self.compressor = codings.svd.SVD(random_sample=False, rank=self.svd_rank,
                                   compress=True)

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
            self.nodes = 1

    def __setstate__(self, state):
        super(SGD_distribute, self).__setstate__(state)


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
                            d_p_new = d_p_new.cpu().numpy().astype(np.float32)
                            coded = self.compressor.encode(d_p_new)
                            #specific coded dic just on CPU
                            tensor_u = torch.from_numpy(coded['u']).float()
                            tensor_s = torch.from_numpy(coded['s']).float()
                            tensor_vT = torch.from_numpy(coded['vT']).float()

                        else:
                            d_p_new = torch.sign(d_p_new)

                        if self.local_rank == 0:
                            if dist.get_rank() == 0:
                                for index, inter_node_group in enumerate(self.inter_node_group_list):
                                    coded_temp = coded.copy()
                                    tensor_u_temp = tensor_u.clone()
                                    tensor_s_temp = tensor_s.clone()
                                    tensor_vT_temp = tensor_vT.clone()

                                    dist.broadcast(tensor_u_temp, self.inter_node_list[index + 1], group = inter_node_group)
                                    dist.broadcast(tensor_s_temp, self.inter_node_list[index + 1], group = inter_node_group)
                                    dist.broadcast(tensor_vT_temp, self.inter_node_list[index + 1], group = inter_node_group)

                                    coded_temp['u'] = tensor_u_temp.numpy().astype(np.float32)
                                    coded_temp['s'] = tensor_s_temp.numpy().astype(np.float32)
                                    coded_temp['vT'] = tensor_vT_temp.numpy().astype(np.float32)

                                    tensor_decoded = self.compressor.decode(coded_temp)
                                    tensor_decoded = tensor_decoded.numpy().astype(np.float32)
                                    d_p_new = d_p_new + tensor_decoded

                            else:
                                dist.broadcast(tensor_u, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1]) 
                                dist.broadcast(tensor_s, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1]) 
                                dist.broadcast(tensor_vT, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1]) 

                                dist.barrier(group = self.all_inter_node_group)

                            if dist.get_rank() == 0:
                                d_p_new = d_p_new / dist.get_world_size()
                                coded = self.compressor.encode(d_p_new)
                                tensor_u = torch.from_numpy(coded['u']).float()
                                tensor_s = torch.from_numpy(coded['s']).float()
                                tensor_vT = torch.from_numpy(coded['vT']).float()

                                dist.barrier(group = self.all_inter_node_group)

                            dist.broadcast(tensor_u, 0, group = self.all_inter_node_group)
                            dist.broadcast(tensor_s, 0, group = self.all_inter_node_group)
                            dist.broadcast(tensor_vT, 0, group = self.all_inter_node_group)

                            coded['u'] = tensor_u.numpy().astype(np.float32)
                            coded['s'] = tensor_s.numpy().astype(np.float32)
                            coded['vT'] = tensor_vT.numpy().astype(np.float32) 

                        tensor_decoded = self.compressor.decode(coded)
                        d_p_new = tensor_decoded.float().cuda()

                    else:
                        # test for one
                        d_p_new = d_p_new.cpu().numpy().astype(np.float32)
                        coded = self.compressor.encode(d_p_new)
                        tensor_decoded = self.compressor.decode(coded)
                        d_p_new = tensor_decoded.float().cuda()


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


