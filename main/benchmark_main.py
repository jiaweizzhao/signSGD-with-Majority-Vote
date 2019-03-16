#!/usr/bin/env python

import argparse, os, shutil, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import torch.multiprocessing as mp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import Signum_SGD
import Imagefolder_train_val
import sys
import tensorboardX

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
#print(model_names)

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--save-dir', type=str, default=Path.cwd(), help='Directory to save logs and models.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=778, type=int,
                        help='seed for initialization')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--sz',       default=224, type=int, help='Size of transformed image.')
    parser.add_argument('--decay-int', default=30, type=int, help='Decay LR by 10 every decay-int epochs')
    parser.add_argument('--warm-up', action='store_true', help='Start warm up at first 5 epochs.')
    parser.add_argument('--prof', dest='prof', action='store_true', help='Only run a few iters for profiling and testing.')
    parser.add_argument('--dist-url', default='env://', type=str, #sync.file
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend')
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--all_reduce', action='store_true', help='Using all_reduce')
    parser.add_argument('--signum', action='store_true', help='Using Signum')
    parser.add_argument('--compress', action='store_true', help='Initiate compression for Signum')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--gpus_per_machine", default=1, type=int)
    parser.add_argument('--test_evaluate', action='store_true', help='Initiate test evaluation')

    #LARC setting
    parser.add_argument('--larc_enable', action='store_true', help='Enable the LARC')
    parser.add_argument("--larc_trust_coefficient", default=0.02, type=float)
    parser.add_argument("--larc_eps", default=1e-8, type=float)
    parser.add_argument('--larc_clip', action='store_true', help='Enable the elip for LARC')

    return parser

cudnn.benchmark = True
args = get_parser().parse_args()

class Time_recorder(object):
    def __init__(self):
        self.time = 0

    def reset(self):
        self.time = 0

    def set(self):
        torch.cuda.synchronize()
        self.begin = time.time()

    def record(self):
        torch.cuda.synchronize()
        self.end = time.time()
        self.time += self.end - self.begin

    def get_time(self):
        return self.time


iter_ptr = 0

train_record = Time_recorder()


def get_loaders(traindir, valdir, use_val_sampler=False, min_scale=0.08, Data_augmentation = True, split_data = False, seed = None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if Data_augmentation:
        train_dataset = Imagefolder_train_val.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), seed = seed, split = split_data, train_data = True)
    else:
        train_dataset = Imagefolder_train_val.ImageFolder(
            traindir,
            transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
            ]), seed = seed, split = split_data, train_data = True)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if split_data:
        val_loader = torch.utils.data.DataLoader(
            Imagefolder_train_val.ImageFolder(traindir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), seed = seed, split = split_data, train_data = False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    return train_loader,val_loader,train_sampler 


def main():

    args.distributed = True

    print("~~epoch\thours\ttop1Accuracy\n")
    start_time = datetime.now()
    if args.distributed:
        os.environ['WORLD_SIZE'] = str(args.world_size)
        dist.init_process_group(backend=args.dist_backend, init_method = args.dist_url, world_size = args.world_size, rank = int(os.environ['RANK']))
        torch.cuda.set_device(args.local_rank)

        if dist.get_rank() == 0:
            print(str(dist.get_world_size()) + ' number of workers is set up!')

    if dist.get_rank() == 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    log_writer = tensorboardX.SummaryWriter(args.save_dir) if dist.get_rank() == 0 else None

    # create model
    model = models.resnet50()

    model = model.cuda()

    #model.para sync
    global param_copy
    param_copy = list(model.parameters())
    for parameter in param_copy:
        dist.broadcast(parameter.data, 0) #group = 0
    if dist.get_rank() == 0:
        print('parameter sync finished')


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = Signum_SGD.SGD_distribute(param_copy, args, log_writer)

    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else: print("=> no checkpoint found at '{}'".format(args.resume))


    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    args.sz = 224

    train_loader,val_loader,train_sampler = get_loaders(traindir, valdir, split_data = not args.test_evaluate, seed = args.seed)

    if args.evaluate: return validate(val_loader, model, criterion, epoch, start_time)

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        if args.distributed:
            train_sampler.set_epoch(epoch)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            train(train_loader, model, criterion, optimizer, epoch, log_writer)

        if args.prof: break
        prec1 = validate(val_loader, model, criterion, epoch, start_time, log_writer)


        if dist.get_rank() == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            '''
            save_checkpoint({
                'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(),
                'best_prec1': best_prec1, 'optimizer' : optimizer.state_dict(),
            }, is_best)
            '''



def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

class data_prefetcher():
    def __init__(self, loader, prefetch=True):
        self.loader,self.prefetch = iter(loader),prefetch
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)

    def next(self):
        if not self.prefetch:
            input,target = next(self.loader)
            return input.cuda(async=True),target.cuda(async=True)

        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def train(train_loader, model, criterion, optimizer, epoch, log_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global iter_ptr


    train_record.set()

    # switch to train mode
    model.train()
    end = time.time()
    torch.cuda.synchronize()
    i = -1
    #while input is not None:
    for input, target in train_loader:
        assert input.size(0) == target.size(0)
        i += 1
        iter_ptr += 1

        if args.prof and (i > 200): break
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(async=True)
        target = target.cuda(async=True)

        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            #reduced_loss = loss.data
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data
        
        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #input, target = prefetcher.next()
        if dist.get_rank() == 0 and i % args.print_freq == 0 and i > 1:
            train_record.record()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Total Training Time {train_time:.3f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, train_time=train_record.get_time()))
            train_record.set()

    if log_writer:
        log_writer.add_scalar('train_iter/top1', top1.get_avg(), iter_ptr)
        log_writer.add_scalar('train_iter/top5', top5.get_avg(), iter_ptr)
        log_writer.add_scalar('train_iter/loss', losses.get_avg(), iter_ptr)
        log_writer.add_scalar('train_iter/batch_time', batch_time.get_avg(), iter_ptr)
        log_writer.add_scalar('train_iter/data_time', data_time.get_avg(), iter_ptr)
        log_writer.add_scalar('train_iter/learning_rate_schedule', args.lr_present, iter_ptr)

        log_writer.add_scalar('train_epoch/top1', top1.get_avg(), epoch)
        log_writer.add_scalar('train_epoch/top5', top5.get_avg(), epoch)
        log_writer.add_scalar('train_epoch/loss', losses.get_avg(), epoch)
        log_writer.add_scalar('train_epoch/learning_rate_schedule', args.lr_present, epoch)

        log_writer.add_scalar('train_time/top1', top1.get_avg(), train_record.get_time())
        log_writer.add_scalar('train_time/top5', top5.get_avg(), train_record.get_time())
        log_writer.add_scalar('train_time/loss', losses.get_avg(), train_record.get_time())  

        if args.larc_enable:
            #add larc_adaptive_lr saving
            laryer_saving_name = ['layer0.conv1.weight', 'layer0.bn1.weight', 'layer1.1.conv1.weight', \
                                'layer2.1.conv1.weight', 'layer3.1.conv1.weight', 'layer4.1.conv1.weight'] #correspond to list laryer_saving in Signum_SGD.py
            for index, layer_lr in enumerate(optimizer.layer_adaptive_lr):
                log_writer.add_scalar('larc_layer_adaptive_lr/' + laryer_saving_name[index], layer_lr, epoch)



def validate(val_loader, model, criterion, epoch, start_time, log_writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global iter_ptr

    model.eval()
    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1

        target = target.cuda(async=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        reduced_loss = reduce_tensor(loss.data)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        

        reduced_prec1 = reduce_tensor(prec1)
        reduced_prec5 = reduce_tensor(prec5)

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if dist.get_rank() == 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

        input, target = prefetcher.next()

    time_diff = datetime.now()-start_time
    if dist.get_rank() == 0:
        print(f'~~{epoch}\t{float(time_diff.total_seconds() / 3600.0)}\t{top5.avg:.3f}\n')
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)) 
        if log_writer:
            log_writer.add_scalar('test_iter/top1', top1.get_avg(), iter_ptr)
            log_writer.add_scalar('test_iter/top5', top5.get_avg(), iter_ptr)
            log_writer.add_scalar('test_iter/loss', losses.get_avg(), iter_ptr)
            log_writer.add_scalar('test_iter/batch_time', batch_time.get_avg(), iter_ptr)

            log_writer.add_scalar('test_epoch/top1', top1.get_avg(), epoch)
            log_writer.add_scalar('test_epoch/top5', top5.get_avg(), epoch)
            log_writer.add_scalar('test_epoch/loss', losses.get_avg(), epoch)

            log_writer.add_scalar('test_time/top1', top1.get_avg(), train_record.get_time())
            log_writer.add_scalar('test_time/top5', top5.get_avg(), train_record.get_time())
            log_writer.add_scalar('test_time/loss', losses.get_avg(), train_record.get_time())  

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{args.save_dir}/model_best.pth.tar')
        

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

def adjust_learning_rate(optimizer, epoch):
    if epoch<5 and args.warm_up: 
        # warmup 5 epochs
        lr = args.lr/(5-epoch)
    else:
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr * (0.1 ** (epoch // args.decay_int))
    args.lr_present = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if dist.get_rank() == 0:
        assert len(pred[1]) == batch_size
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if dist.get_rank() == 0:
        assert len(correct[1]) == batch_size

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt

if __name__ == '__main__': 
    main()