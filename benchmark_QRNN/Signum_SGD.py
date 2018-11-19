#coding=utf-8
import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
import compressor
import time
import os
import math
#Signum with majority vote


class Time_recorder(object):
    def __init__(self):
        self.time = 0
        self.begin = 0

    def reset(self):
        self.time = 0
        #pass

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

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay = 0, compression_buffer = False, all_reduce = False ,local_rank = 0, gpus_per_machine = 1 , single_worker = False, **kwargs):
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
        self.single_worker = single_worker
        self.gpus_per_machine = gpus_per_machine
        #gpus_per_machine = torch.cuda.device_count()
        print('The number of GPUs is', gpus_per_machine)
        print('Single worker', single_worker)

        self.MB = 1024 * 1024
        self.bucket_size = 200 * self.MB

        if self.compression_buffer and not self.single_worker:

            self.compressor = compressor.compressor(using_cuda = True, local_rank = 1)
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


        #add time record
        self.all_time = Time_recorder()
        self.all_reduce_time = Time_recorder()
        self.compression_time = Time_recorder()
        self.uncompression_time = Time_recorder()
        self.broadcast_time = Time_recorder()
        self.majority_vote_time = Time_recorder()
        self.compress_all_time = Time_recorder()
        self.gather_all_time = Time_recorder()
        self.calculate_all_time = Time_recorder()
        self.broadcast_all_time = Time_recorder()
        self.update_para_time = Time_recorder()
        self.bucketing_time = Time_recorder()
        self.debucketing_time = Time_recorder()


    def __setstate__(self, state):
        super(SGD_distribute, self).__setstate__(state)

    def get_time_result(self):
        if dist.get_rank() == 0:
            print('all_time',self.all_time.get_time())
            print('compress_in_each_device_time',self.compress_all_time.get_time())
            print('gather_time',self.gather_all_time.get_time())
            print('calculate_in_PS_time',self.calculate_all_time.get_time())
            print('broadcast__to_all_time',self.broadcast_all_time.get_time())
            print('all_reduce_time',self.all_reduce_time.get_time())
            print('first_compression_time',self.compression_time.get_time())
            print('last_decompression_time',self.uncompression_time.get_time())
            print('broadcast_time',self.broadcast_time.get_time())
            print('majority_vote_time',self.majority_vote_time.get_time())
            print('update_para_time',self.update_para_time.get_time())
            print('bucketing_time',self.bucketing_time.get_time())
            print('debucketing_time',self.debucketing_time.get_time())
            self.compressor.get_time_result()

    def reset_time(self):
        self.all_time.reset()
        self.all_reduce_time.reset()
        self.compression_time.reset()
        self.uncompression_time.reset()
        self.broadcast_time.reset()
        self.majority_vote_time.reset()
        self.compress_all_time.reset()
        self.gather_all_time.reset()
        self.calculate_all_time.reset()
        self.broadcast_all_time.reset()
        self.update_para_time.reset()
        self.bucketing_time.reset()
        self.debucketing_time.reset()

        self.compressor.reset_time()


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
                '''
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                '''
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

            #torch.cuda.init()
            #torch.cuda.empty_cache()

            if not self.single_worker:

                self.all_time.set()
                self.bucketing_time.set()
                #start bucketing
                dev_grads_buckets = _take_tensors(all_grads, self.bucket_size)
                self.bucketing_time.record()
                for dev_grads in dev_grads_buckets:


                    self.bucketing_time.set()
                    d_p_new = _flatten_dense_tensors(dev_grads)
                    #print('the size of each bucket',d_p_new.size())
                    #os._exit(0)
                    self.bucketing_time.record()



                    #d_p_new = d_p.clone()
                    if self.all_reduce:
                        self.all_reduce_time.set()
                        #torch.cuda.synchronize()
                        #d_p_new = torch.sign(d_p_new)
                        dist.all_reduce(d_p_new, group = 0) #self.all_gpu

                        #use boradcast_gather

                        
                        #take the sign to test
                        #d_p_new = torch.sign(d_p_new)
                        #torch.cuda.synchronize()
                        self.all_reduce_time.record()
                    else:
                        #print('once')
                        self.compress_all_time.set()
                        self.all_reduce_time.set()
                        #torch.cuda.synchronize()

                        if self.gpus_per_machine > 1:
                            dist.all_reduce(d_p_new, group = self.intra_node_group_list[self.nodes_rank])
                            dist.barrier(group = self.all_gpu)
                        self.all_reduce_time.record()

                        #leave compression
                        if self.nodes > 1:

                            self.compression_time.set()
                            ##torch.cuda.synchronize()
                            if self.compression_buffer:
                                d_p_new, tensor_size = self.compressor.compress(d_p_new)
                            else:
                                d_p_new = torch.sign(d_p_new)

                            ##torch.cuda.synchronize()
                            self.compression_time.record()
                            self.compress_all_time.record()
                            self.gather_all_time.set()
                            if self.local_rank == 0:
                                if dist.get_rank() == 0:
                                    d_p_new_list = []
                                    for index, inter_node_group in enumerate(self.inter_node_group_list):
                                        #print('gather', inter_node_list[index + 1])
                                        d_p_temp = d_p_new.clone()
                                        self.broadcast_time.set()
                                        dist.broadcast(d_p_temp, self.inter_node_list[index + 1], group = inter_node_group)
                                        self.broadcast_time.record()
                                        d_p_new_list.append(d_p_temp)
                                else:
                                    self.broadcast_time.set()
                                    dist.broadcast(d_p_new, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1])
                                    self.broadcast_time.record()
                                    #print(dist.get_rank(), 'finish broadcast')
                                    
                                    dist.barrier(group = self.all_inter_node_group)
                                self.gather_all_time.record()
                                self.calculate_all_time.set()


                                if dist.get_rank() == 0:

                                    self.majority_vote_time.set()
                                    if self.compression_buffer:
                                        d_p_new_list.append(d_p_new) #count itself
                                        d_p_new = self.compressor.majority_vote(d_p_new_list)
                                    else:
                                        for d_p_temp in d_p_new_list:
                                            d_p_new.add_(d_p_temp)
                                        d_p_new = d_p_new / self.nodes


                                    ##torch.cuda.synchronize()
                                    self.majority_vote_time.record()
                                    dist.barrier(group = self.all_inter_node_group)
                                self.calculate_all_time.record()
                                self.broadcast_all_time.set()
                                self.broadcast_time.set()
                                dist.broadcast(d_p_new, 0, group = self.all_inter_node_group)
                                self.broadcast_time.record()

                                #dist.barrier(group = self.all_inter_node_group)

                            #broadcast to all
                            #print('start broadcast')
                            #self.broadcast_time.set()
                            dist.broadcast(d_p_new, self.local_dst_in_global, group = self.intra_node_group_list[self.nodes_rank])
                            #self.broadcast_time.record()
                            self.uncompression_time.set()
                            ##torch.cuda.synchronize()
                            if self.compression_buffer:
                                d_p_new = self.compressor.uncompress(d_p_new, tensor_size)

                            #torch.cuda.synchronize()
                            self.uncompression_time.record()
                            self.broadcast_all_time.record()
                            #os._exit(0)


                    self.debucketing_time.set()
                    #unflatten
                    dev_grads_new = _unflatten_dense_tensors(d_p_new,dev_grads)
                    for grad, reduced in zip(dev_grads, dev_grads_new):
                        grad.copy_(reduced)
                    self.debucketing_time.record()

                self.all_time.record()

            self.update_para_time.set()
            #torch.cuda.synchronize()
            for p in group['params']:
                if weight_decay != 0:
                    p.grad.data.add_(weight_decay, p.data)
                if self.single_worker and self.compression_buffer:
                    p.data.add_(-group['lr'], torch.sign(p.grad.data))
                else:
                    p.data.add_(-group['lr'], p.grad.data)

            #torch.cuda.synchronize()
            self.update_para_time.record()




        return loss


class Adam_distribute(Optimizer):
    """Implements Adam_distribute algorithm.

    It has been proposed in `Adam_distribute: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam_distribute and Beyond`_

    .. _Adam_distribute: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam_distribute and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, compression_buffer = False, all_reduce = True ,local_rank = 0, gpus_per_machine = 1, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_distribute, self).__init__(params, defaults)


        #custom code
        self.compression_buffer = compression_buffer
        self.all_reduce = all_reduce
        #gpus_per_machine = torch.cuda.device_count()
        print('The number of GPUs is', gpus_per_machine)

        self.MB = 1024 * 1024
        self.bucket_size = 200 * self.MB
        self.all_reduce_time = Time_recorder()

    def __setstate__(self, state):
        super(Adam_distribute, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    def get_time_result(self):
        if dist.get_rank() == 0:
            print('all_reduce_time',self.all_reduce_time.get_time())
            #self.compressor.get_time_result()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # collect gradients first
            all_grads = []

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                all_grads.append(d_p)
            dev_grads_buckets = _take_tensors(all_grads, self.bucket_size)
            for dev_grads in dev_grads_buckets:
                d_p_new = _flatten_dense_tensors(dev_grads)
                if self.all_reduce:
                    self.all_reduce_time.set()
                    dist.all_reduce(d_p_new, group = 0)
                    self.all_reduce_time.record()

                dev_grads_new = _unflatten_dense_tensors(d_p_new,dev_grads)
                for grad, reduced in zip(dev_grads, dev_grads_new):
                    grad.copy_(reduced)


            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Adam_distribute does not support sparse gradients, please consider SparseAdam_distribute instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


