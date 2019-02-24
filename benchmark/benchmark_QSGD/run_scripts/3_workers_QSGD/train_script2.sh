#repeat experiment
cd imagenet-fast_jiawei
cd imagenet_nv
mkdir result
cd result
mkdir QSGD_final_multi_level_test
cd ..
for repeat_num in 1
do
    cd result
    cd QSGD_final_multi_level_test
    mkdir QSGD_normal_norm=$repeat_num
    cd QSGD_normal_norm=$repeat_num
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 90 --save-dir ./result/QSGD_final_multi_level_test/QSGD_normal_norm=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --qsgd_level 2 --enable_max --all_reduce \
    --dist-url tcp://ec2-52-11-11-115.us-west-2.compute.amazonaws.com:1235
done