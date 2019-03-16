# signSGD-with-Majority-Vote

This repository contains the code used for paper:
+ [signSGD with Majority Vote is Commnunication Efficient and Byzantine Fault Tolerant](https://openreview.net/forum?id=BJxhijAcY7)

This code was originally forked from the [End to end ImageNet training](https://github.com/fastai/imagenet-fast).


## Pre-installation

### Downloading ImageNet
1. You can download ImageNet from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).
2. You can download from our S3 bucket (s3://signum-majority-vote/dataset/ILSVRC.tar) (reproduction purpose only).

### C++ extension installation 
1. Put the folder 'main/bit2byte-extension' to the directory of the PyTorch source code
2. Execute this command on the directory of 'bit2byte-extension'  
`python setup.py install`  
You can find more information about C++ extension in [PyTorch documentation](https://pytorch.org/tutorials/advanced/cpp_extension.html#using-your-extension)

## Experiments

**Note:** You have to execute following commands in each instance.

### ImageNet Benchmark

Execute following commands on the directory of 'main'

#### Training Signum

+ `ulimit -n 1000000`
`sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=[number of instances] --node_rank=[rank of instance] --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.0001 \
--epochs 80 --save-dir ./ --world-size [number of instances] --print-freq 200 --compress --dist_backend gloo --weight-decay 1e-4 --momentum 0.9 --warm-up \
--dist-url [parameter sever's url]`

#### Training Vanilla SGD


+ `ulimit -n 1000000`
`sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=[number of instances] --node_rank=[rank of instance] --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 \
--epochs 80 --save-dir ./ --world-size [number of instances] --print-freq 200 --all_reduce --dist_backend nccl --weight-decay 0.1 --momentum 0.9 --warm-up \
--dist-url [parameter sever's url]`

### QRNN Benchmark

Execute following commands on the directory of 'benchmark/QRNN'

#### Training Signum

+ `/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=[number of instances] --node_rank=[rank of instance] --master_addr="0.0.0.0" \
--master_port=1235 main_signum.py --epochs 12 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 240 \
--optimizer signum --lr 1e-3 --momentum 0.5 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
--world-size [number of instances] --dist-url [parameter sever's url] \
--save-dir ./ --distributed --multi_gpu --momentun_warm_up
`

#### Training Adam

+ `/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=[number of instances] --node_rank=[rank of instance] --master_addr="0.0.0.0" \
--master_port=1235 main_signum.py --epochs 12 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 240 \
--optimizer adam --lr 1e-3 --momentum 0.5 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
--world-size [number of instances] --dist-url [parameter sever's url] \
--save-dir ./ --distributed --multi_gpu --momentun_warm_up
`

### QSGD Benchmark

Execute following commands on the directory of 'benchmark/QSGD'

#### Training Signum

+ `ulimit -n 1000000`
`sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=[number of instances] --node_rank=[rank of instance] --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-4 --seed 1 \
--epochs 90 --save-dir ./ --world-size [number of instances] --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method Signum \
--dist-url [parameter sever's url]
`

#### Training QSGD

+ `ulimit -n 1000000`
`sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=[number of instances] --node_rank=[rank of instance] --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed 1 \
--epochs 90 --save-dir ./ --world-size [number of instances] --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method QSGD --qsgd_level [the level of QSGD] [--enable_max, if enable max_norm] --all_reduce \
--dist-url [parameter sever's url]
`

### Krum Benchmark

Execute following commands on the directory of 'benchmark/Krum'

#### Training Signum

+ `ulimit -n 1000000`
`sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=[number of instances] --node_rank=[rank of instance] --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-3 \
--epochs 90 --save-dir ./ --world-size 7 --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method Signum \
--enable_adversary --adversary_num [the number of adversaries] [--enable_minus_adversary, enable minus adversary or it will be random one] \
--dist-url [parameter sever's url]
`

#### Training Krum

+ `ulimit -n 1000000`
`sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=[number of instances] --node_rank=[rank of instance] --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-1 \
--epochs 90 --save-dir ./ --world-size 7 --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method Signum \
--enable_krum --krum_f [the number of F] --enable_adversary --adversary_num [the number of adversaries] \
--dist-url [parameter sever's url]
`


