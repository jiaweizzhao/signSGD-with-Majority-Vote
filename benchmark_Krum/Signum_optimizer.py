#coding=utf-8
import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
import compressor
import time
import os
import byzantine_utils
#Signum with majority vote


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
        self.use_majority_vote = not args.disable_majority_vote
        self.enable_krum = args.enable_krum
        self.krum_f = args.krum_f
        self.enable_adversary = args.enable_adversary
        self.adversary_num = args.adversary_num
        self.enable_minus_adversary = args.enable_minus_adversary

        if self.enable_krum:
            self.compression_buffer = True
            self.all_reduce = False
            self.use_majority_vote = True
            self.enable_minus_adversary = False


        self.MB = 1024 * 1024
        self.bucket_size = 100 * self.MB

        if self.compression_buffer:

            self.compressor = compressor.compressor(using_cuda = True, local_rank = local_rank)
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
                    #become adversary
                    if self.enable_adversary:
                        if dist.get_rank() <= (self.adversary_num - 1):
                            if self.enable_minus_adversary:
                                d_p_new = - d_p_new
                            else:
                                d_p_new = torch.randn(d_p_new.size()).type_as(d_p_new)
                    if self.nodes > 1:
                        if self.enable_krum:
                            #send full tensor
                            d_p_new = d_p_new
                        elif self.compression_buffer:
                            d_p_new_origial = d_p_new
                            d_p_new, tensor_size = self.compressor.compress(d_p_new)
                        else:
                            d_p_new = torch.sign(d_p_new)


                        if self.local_rank == 0:
                            if dist.get_rank() == 0:
                                if self.use_majority_vote:
                                    d_p_new_list = []
                                    for index, inter_node_group in enumerate(self.inter_node_group_list):
                                        d_p_temp = d_p_new.clone()
                                        dist.broadcast(d_p_temp, self.inter_node_list[index + 1], group = inter_node_group)
                                        d_p_new_list.append(d_p_temp)
                                else:
                                    for index, inter_node_group in enumerate(self.inter_node_group_list):
                                        d_p_temp = d_p_new.clone()
                                        dist.broadcast(d_p_temp, self.inter_node_list[index + 1], group = inter_node_group)
                                        d_p_uncompressed = self.compressor.uncompress(d_p_temp, tensor_size)
                                        d_p_new_origial += d_p_uncompressed
                                    d_p_new, tensor_size = self.compressor.compress(d_p_new_origial)

                            else:
                                dist.broadcast(d_p_new, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1])                                
                                dist.barrier(group = self.all_inter_node_group)

                            if dist.get_rank() == 0:
                                if self.compression_buffer:
                                    if self.enable_krum:
                                        d_p_new_list.append(d_p_new)
                                        #byzantine_setting_krum
                                        d_p_new = byzantine_utils.krum(d_p_new_list, self.krum_f, multi=True) #.type_as(d_p_new)
                                        d_p_new = d_p_new / (len(d_p_new_list) - self.krum_f)
                                    elif self.use_majority_vote:
                                        d_p_new_list.append(d_p_new) #count itself
                                        d_p_new = self.compressor.majority_vote(d_p_new_list)
                                    else:
                                        pass
                                else:
                                    for d_p_temp in d_p_new_list:
                                        d_p_new.add_(d_p_temp)
                                    d_p_new = d_p_new / self.nodes
                                dist.barrier(group = self.all_inter_node_group)

                            dist.broadcast(d_p_new, 0, group = self.all_inter_node_group)

                        if self.enable_krum:
                            d_p_new = d_p_new
                        elif self.compression_buffer:
                            d_p_new = self.compressor.uncompress(d_p_new, tensor_size)
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


