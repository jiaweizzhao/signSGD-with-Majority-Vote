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
2. Execute this command on the directory of 'main/bit2byte-extension'  
`python setup.py install`  
You can find more information about C++ extension in [PyTorch documentation](https://pytorch.org/tutorials/advanced/cpp_extension.html#using-your-extension)

## Experiments

**Note:** You have to execute following commands in each instance.

### ImageNet Benchmark

Execute following commands on the directory of 'main'

#### Training Signum

+ `ulimit -n 1000000`
`sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=[rank of instance] --nnodes=[number of instances] --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.0001 \
--epochs 80 --save-dir ./ --world-size [number of instances] --print-freq 200 --compress --dist_backend gloo --weight-decay 0.1 --momentum 0.9 \
--dist-url [parameter sever's url]`

### Training Vanilla SGD


+ `ulimit -n 1000000`
`sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=[rank of instance] --nnodes=[number of instances] --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 \
--epochs 80 --save-dir ./ --world-size [number of instances] --print-freq 200 --all_reduce --dist_backend nccl --weight-decay 0.0001 --momentum 0.9 \
--dist-url [parameter sever's url]`

### QRNN Benchmark

Execute following commands on the directory of 'benchmark/QRNN'

#### Training Signum

#### Training Adam

### QSGD Benchmark

Execute following commands on the directory of 'benchmark/QSGD'

#### Training Signum

#### Training QSGD

### Krum Benchmark

Execute following commands on the directory of 'benchmark/Krum'

#### Training Signum

#### Training Krum


