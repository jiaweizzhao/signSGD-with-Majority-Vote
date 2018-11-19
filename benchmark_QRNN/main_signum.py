import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import argparse, os, shutil, time, warnings
from datetime import datetime
from pathlib import Path
import data
import model
import torch.distributed as dist
import tensorboardX
import sys
import Signum_SGD
import copy

from utils import batchify, get_batch, repackage_hidden, batchify_distributed

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=778,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument("--local_rank", type=int)
parser.add_argument('--world-size', default=1, type=int,
                    help='Number of GPUs to use. Can either be manually set ' +
                    'or automatically set by using \'python -m multiproc\'.')
parser.add_argument('--dist-url', default='env://', type=str, #sync.file
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--save-dir', type=str, default=Path.cwd(), help='Directory to save logs and models.')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='the number of momentun')
parser.add_argument('--distributed', action='store_true',
                    help='use distributed')
parser.add_argument('--tuning_mode', action='store_true',
                    help='tuning')
parser.add_argument('--momentun_warm_up', action='store_true',
                    help='tuning')
parser.add_argument('--multi_gpu', action='store_true',
                    help='tuning')
args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

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

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

#distributed initialization
if args.distributed:
    print('start dist inital')
    os.environ['WORLD_SIZE'] = str(args.world_size)
    #print(int(os.environ['RANK']))
    #dist.init_process_group(backend=args.dist_backend, init_method = args.dist_url)
    dist.init_process_group(backend=args.dist_backend, init_method = args.dist_url, world_size = args.world_size, rank = int(os.environ['RANK']))
    print(str(dist.get_world_size()) + ' number of workers is set up!')
    torch.cuda.set_device(args.local_rank)
    print('inital finished')

if args.distributed:
    log_writer = tensorboardX.SummaryWriter(args.save_dir) if dist.get_rank() == 0 else None
else:
    log_writer = tensorboardX.SummaryWriter(args.save_dir)

train_record = Time_recorder()
iter_ptr = 0

eval_batch_size = 10
test_batch_size = 1

if args.distributed:
    train_data = batchify_distributed(corpus.train, args.batch_size, args, 0)
else:
    train_data = batchify(corpus.train, args.batch_size, args, 0)
print('train data size', train_data.size())

#train_data = train_data.to(torch.device('cuda:1'))


val_data = batchify(corpus.valid, eval_batch_size, args, 0)
test_data = batchify(corpus.test, test_batch_size, args, 0)


###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)

###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###


if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()


###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


if args.distributed:
    #model.para sync custom
    for parameter in params: #model.parameters():
        dist.broadcast(parameter.data, 0, group = 0)
    if dist.get_rank() == 0:
        print('parameter sync finished')

if args.multi_gpu:
    model = torch.nn.DataParallel(model,device_ids = [0,1,2,3,4,5,6,7], dim=1)
    criterion = torch.nn.DataParallel(criterion,device_ids = [0,1,2,3,4,5,6,7])
else:
    model = torch.nn.DataParallel(model,device_ids = [0], dim=1)
    criterion = torch.nn.DataParallel(criterion,device_ids = [0])   
    #model.to()

#intial torch.nn.DataParallel



###############################################################################
# Training code
###############################################################################
'''
def evaluate(data_source, batch_size=10):
    
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.module.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.module.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model.module(input = data, hidden = hidden)
        criterion.module.replicate_weight_and_bias(model.module.decoder.weight,model.module.decoder.bias)
        raw_loss = criterion(hiddens = output, targets = targets)
        raw_loss = torch.mean(input=raw_loss, dim=0)
        total_loss += len(data) * raw_loss.data

        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)
'''
    
    
    #return 10


def evaluate(model, criterion, data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    model_now = model.module
    criterion_now = criterion.module
    if args.model == 'QRNN': model_now.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model_now.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model_now(data, hidden)
        criterion_now.replicate_weight_and_bias(torch.nn.Parameter(model.module.decoder.weight),torch.nn.Parameter(model.module.decoder.bias))
        total_loss += len(data) * criterion_now(hiddens = output, targets = targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train():
    communication_cost = Time_recorder()
    calculation_cost = Time_recorder()
    epoch_cost = Time_recorder()
    epoch_cost.set()
    train_record.set()
    global iter_ptr

    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.module.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.module.init_hidden(args.batch_size)
    batch, i = 0, 0
    iter_num = 0
    save_num = 0

    while i < train_data.size(0) - 1 - 1:
        #print('iter: ',i)
        '''
        if iter_ptr == 5:
            optimizer.get_time_result()
        '''
        iter_ptr += 1
        calculation_cost.set()
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        #data = data.to(torch.device('cuda:1'))
        #hidden = hidden.to(torch.device('cuda:1'))

        output, hidden, rnn_hs, dropped_rnn_hs = model(input = data, hidden = hidden, return_h=True)
        #move out
        output = output.view(output.size(0)*output.size(1), output.size(2))

        criterion.module.replicate_weight_and_bias(model.module.decoder.weight,model.module.decoder.bias)
        #criterion.replicate_weight_and_bias(model.module.decoder.weight,model.module.decoder.bias)

        raw_loss = criterion(hiddens = output, targets = targets)
        raw_loss = torch.mean(input=raw_loss, dim=0)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        calculation_cost.record()
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        communication_cost.set()

        #delete something
        del data
        del targets

        optimizer.step()
        communication_cost.record()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        
        if args.momentun_warm_up and iter_ptr >= 5481:
            args.momentun_warm_up = False
            args.momentum = 0.9
            momentum = args.momentum
            lr_new = 1e-4
            print('momentum change to',momentum)
            optimizer.param_groups[0]['momentum'] = momentum
            optimizer.param_groups[0]['lr'] = lr_new

        '''
        
        if args.momentun_warm_up and iter_ptr == 2500:
            momentum = 0.999
            lr_new = 1e-4
            print('momentum change to',momentum)
            optimizer.param_groups[0]['momentum'] = momentum
            optimizer.param_groups[0]['lr'] = lr_new
        '''
        '''
        if iter_ptr ==5:
            save_num = time.time() - start_time
        if iter_ptr ==15:
            print(time.time() - start_time - save_num)
        '''
        

        if batch % args.log_interval == 0 and batch > 0:
            epoch_cost.record()
            train_record.record()
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))

            
            model_evalue = copy.deepcopy(model)
            criterion_evalue = copy.deepcopy(criterion)
            #model_evalue = model_evalue.cuda()
            #test evaluate
            test_loss = evaluate(model_evalue, criterion_evalue, test_data, test_batch_size)
            if log_writer:
                log_writer.add_scalar('test_iter/perplexity', math.exp(test_loss), iter_ptr)
                log_writer.add_scalar('test_iter/loss', test_loss, iter_ptr)
                log_writer.add_scalar('test_iter/bpc', test_loss / math.log(2), iter_ptr)

                log_writer.add_scalar('test_time/perplexity', math.exp(test_loss), train_record.get_time())
                log_writer.add_scalar('test_time/loss', test_loss, train_record.get_time())
                log_writer.add_scalar('test_time/bpc', test_loss / math.log(2), train_record.get_time())   

            #validation evaluate
            val_loss2 = evaluate(model_evalue, criterion_evalue, val_data)

            if log_writer:
                log_writer.add_scalar('val_iter/perplexity', math.exp(val_loss2), iter_ptr)
                log_writer.add_scalar('val_iter/loss', val_loss2, iter_ptr)
                log_writer.add_scalar('val_iter/bpc', val_loss2 / math.log(2), iter_ptr)

                log_writer.add_scalar('val_time/perplexity', math.exp(val_loss2), train_record.get_time())
                log_writer.add_scalar('val_time/loss', val_loss2, train_record.get_time())
                log_writer.add_scalar('val_time/bpc', val_loss2 / math.log(2), train_record.get_time())   

            del model_evalue, criterion_evalue
            

            if log_writer:
                iter_num += 1
                log_writer.add_scalar('train_iter/learning_rate', optimizer.param_groups[0]['lr'], iter_ptr)
                #log_writer.add_scalar('train_iter/momentum', optimizer.param_groups[0]['momentum'], iter_ptr)
                log_writer.add_scalar('train_iter/perplexity', math.exp(cur_loss), iter_ptr)
                log_writer.add_scalar('train_iter/loss', cur_loss, iter_ptr)
                log_writer.add_scalar('train_iter/bpc', cur_loss / math.log(2), iter_ptr)
                log_writer.add_scalar('train_iter/ms_batch', elapsed * 1000 / args.log_interval, iter_ptr)
                log_writer.add_scalar('train_iter/epoch_communication_cost', communication_cost.get_time(), iter_ptr)
                log_writer.add_scalar('train_iter/epoch_calculation_cost', calculation_cost.get_time(), iter_ptr)
                log_writer.add_scalar('train_iter/epoch_full_cost', epoch_cost.get_time(), iter_ptr)
                
                log_writer.add_scalar('train_epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)
                log_writer.add_scalar('train_epoch/perplexity', math.exp(cur_loss), epoch)
                log_writer.add_scalar('train_epoch/loss', cur_loss, epoch)
                log_writer.add_scalar('train_epoch/bpc', cur_loss / math.log(2), epoch)
                
                log_writer.add_scalar('train_time/perplexity', math.exp(cur_loss), train_record.get_time())
                log_writer.add_scalar('train_time/loss', cur_loss, train_record.get_time())
                log_writer.add_scalar('train_time/bpc', cur_loss / math.log(2), train_record.get_time())

                if args.tuning_mode and iter_ptr >= 1300:
                    sys.exit(0)

            total_loss = 0
            start_time = time.time()
            train_record.set()
            epoch_cost.set()

        ###
        batch += 1
        i += seq_len

    epoch_cost.record()
    train_record.record()


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    #custom code
    if args.optimizer == 'sgd_distribute':
        optimizer = Signum_SGD.SGD_distribute(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay, local_rank = args.local_rank, compression_buffer = False, all_reduce = True)
    if args.optimizer == 'adam':
        if args.distributed:
            optimizer = Signum_SGD.Adam_distribute(params, lr=args.lr, weight_decay=args.wdecay, local_rank = args.local_rank)
        else:
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'signum':
            optimizer = Signum_SGD.SGD_distribute(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay, local_rank = args.local_rank, compression_buffer = True, all_reduce = False)       

    #print('momentum set to 0')
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        #train_record.set()
        train()
        #custom shuffle the data each epoch
        del train_data
        if args.distributed:
            train_data = batchify_distributed(corpus.train, args.batch_size, args, epoch)
        else:
            train_data = batchify(corpus.train, args.batch_size, args, epoch)
        #train_record.record()
        #custom test record
        '''
        test_loss = evaluate(test_data, test_batch_size)
        if log_writer:
            log_writer.add_scalar('test_iter/perplexity', math.exp(test_loss), iter_ptr)
            log_writer.add_scalar('test_iter/loss', test_loss, iter_ptr)
            log_writer.add_scalar('test_iter/bpc', test_loss / math.log(2), iter_ptr)

            log_writer.add_scalar('test_time/perplexity', math.exp(test_loss), train_record.get_time())
            log_writer.add_scalar('test_time/loss', test_loss, train_record.get_time())
            log_writer.add_scalar('test_time/bpc', test_loss / math.log(2), train_record.get_time())
        '''   

        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(model, criterion, val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)
            '''
            if log_writer:
                log_writer.add_scalar('val_iter/perplexity', math.exp(val_loss2), iter_ptr)
                log_writer.add_scalar('val_iter/loss', val_loss2, iter_ptr)
                log_writer.add_scalar('val_iter/bpc', val_loss2 / math.log(2), iter_ptr)

                log_writer.add_scalar('val_time/perplexity', math.exp(val_loss2), train_record.get_time())
                log_writer.add_scalar('val_time/loss', val_loss2, train_record.get_time())
                log_writer.add_scalar('val_time/bpc', val_loss2 / math.log(2), train_record.get_time())
            '''            

            if val_loss2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(model, criterion, val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            '''
            if log_writer:
                log_writer.add_scalar('val_iter/perplexity', math.exp(val_loss), iter_ptr)
                log_writer.add_scalar('val_iter/loss', val_loss, iter_ptr)
                log_writer.add_scalar('val_iter/bpc', val_loss / math.log(2), iter_ptr)

                log_writer.add_scalar('val_time/perplexity', math.exp(val_loss), train_record.get_time())
                log_writer.add_scalar('val_time/loss', val_loss, train_record.get_time())
                log_writer.add_scalar('val_time/bpc', val_loss / math.log(2), train_record.get_time()) 
            '''
            '''
            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss
            '''
            '''
            switch off this ASGD
            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            '''
            if epoch in args.when or epoch == 20:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
