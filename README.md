# signSGD-with-Majority-Vote

This repository contains the code used for paper:
+ [signSGD with Majority Vote is Commnunication Efficient and Byzantine Fault Tolerant](https://openreview.net/forum?id=BJxhijAcY7)

This code was originally forked from the [End to end ImageNet training](https://github.com/fastai/imagenet-fast).

## Experiments

### Downloading ImageNet
1. Run pre-scripts/download_kaggle_imagenet.sh to download the ImageNet dataset
2. Run pre-scripts/imagenet_formatting.sh to preprocess the images

### C++ extension installation 
1. Put the folder 'bit2byte-extension' to the directory of PyTorch source code
2. Execute this command on the directory of 'bit2byte-extension'  
`python setup.py install`  
You can find more information about C++ extension in [PyTorch documentation](https://pytorch.org/tutorials/advanced/cpp_extension.html#using-your-extension)

### Training Signum


**Note:** You have to execute this command in each instance.

+ `ulimit -n 1000000`
`sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=[rank of instance] --nnodes=[number of instances] --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.0001 \
--epochs 80 --save-dir ./ --world-size [number of instances] --print-freq 200 --compress --dist_backend gloo --weight-decay 0.1 --momentum 0.9 \
--dist-url [parameter sever's url]`


### Training SGD
**Note:** You have to execute this command in each instance.

+ `ulimit -n 1000000`
`sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=[rank of instance] --nnodes=[number of instances] --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 \
--epochs 80 --save-dir ./ --world-size [number of instances] --print-freq 200 --all_reduce --dist_backend nccl --weight-decay 0.0001 --momentum 0.9 \
--dist-url [parameter sever's url]`
