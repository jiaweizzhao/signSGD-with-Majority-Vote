#formal experiment 
#7 workers, 128 batch-size
cd imagenet-fast_jiawei
cd imagenet_nv
mkdir result
cd result
mkdir 7_workers_Cifar10_adversary_comparision
cd ..
for adversaries in 0
do
    cd result
    cd 7_workers_Cifar10_adversary_comparision
    mkdir 7_Signum_minus_adversaries=$adversaries
    cd 7_Signum_minus_adversaries=$adversaries
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=7 --node_rank=5 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-3 \
    --epochs 90 --save-dir ./result/7_workers_Cifar10_adversary_comparision/7_Signum_minus_adversaries=$adversaries --world-size 7 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method Signum \
    --enable_adversary --adversary_num $adversaries --enable_minus_adversary \
    --dist-url tcp://ec2-35-167-102-54.us-west-2.compute.amazonaws.com:1235 

    cd result
    cd 7_workers_Cifar10_adversary_comparision
    mkdir 7_Signum_random_adversaries=$adversaries
    cd 7_Signum_random_adversaries=$adversaries
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000   
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=7 --node_rank=5 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-3 \
    --epochs 90 --save-dir ./result/7_workers_Cifar10_adversary_comparision/7_Signum_random_adversaries=$adversaries --world-size 7 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method Signum \
    --enable_adversary --adversary_num $adversaries \
    --dist-url tcp://ec2-35-167-102-54.us-west-2.compute.amazonaws.com:1235

    cd result
    cd 7_workers_Cifar10_adversary_comparision
    mkdir 7_Krum_random_adversaries=$adversaries
    cd 7_Krum_random_adversaries=$adversaries
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000  
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=7 --node_rank=5 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-1 \
    --epochs 90 --save-dir ./result/7_workers_Cifar10_adversary_comparision/7_Krum_random_adversaries=$adversaries --world-size 7 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method Signum \
    --enable_krum --krum_f 2 --enable_adversary --adversary_num $adversaries \
    --dist-url tcp://ec2-35-167-102-54.us-west-2.compute.amazonaws.com:1235

done