
cd imagenet-fast_jiawei
cd imagenet_nv
mkdir result
cd result
mkdir Signum_and_QSGD_final_short_all_reduce
cd ..

for repeat_num in 1 2 3 
do
    cd result
    cd Signum_and_QSGD_final_short_all_reduce
    mkdir 3_Signum_repeat_num=$repeat_num=NCCL_test
    cd 3_Signum_repeat_num=$repeat_num=NCCL_test
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 90 --save-dir ./result/Signum_and_QSGD_final_short_all_reduce/3_Signum_repeat_num=$repeat_num=NCCL_test --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method Signum --all_reduce \
    --dist-url tcp://ec2-18-236-214-72.us-west-2.compute.amazonaws.com:1235
done