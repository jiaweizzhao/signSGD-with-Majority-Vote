
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
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 90 --save-dir ./result/Signum_and_QSGD_final_short_all_reduce/3_Signum_repeat_num=$repeat_num=NCCL_test --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method Signum --all_reduce \
    --dist-url tcp://ec2-18-236-214-72.us-west-2.compute.amazonaws.com:1235
done

'''
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
    mkdir 3_QSGD_bidirection_max_norm_repeat_num=$repeat_num
    cd 3_QSGD_bidirection_max_norm_repeat_num=$repeat_num
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 90 --save-dir ./result/Signum_and_QSGD_final_short_all_reduce/3_QSGD_bidirection_max_norm_repeat_num=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --bidirection_compress --enable_max --all_reduce \
    --dist-url tcp://ec2-34-221-38-127.us-west-2.compute.amazonaws.com:1235
done
'''


'''
cd imagenet-fast_jiawei
cd imagenet_nv
mkdir result
cd result
mkdir Signum_and_QSGD_final_short
cd ..
for repeat_num in 1 2 3 
do
    cd result
    cd Signum_and_QSGD_final_short
    mkdir 3_QSGD_bidirection_max_norm_repeat_num=$repeat_num
    cd 3_QSGD_bidirection_max_norm_repeat_num=$repeat_num
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 35 --save-dir ./result/Signum_and_QSGD_final_short/3_QSGD_bidirection_max_norm_repeat_num=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --bidirection_compress --enable_max \
    --dist-url tcp://ec2-34-221-38-127.us-west-2.compute.amazonaws.com:1235
done

for repeat_num in 1 2 3 
do
    cd result
    cd Signum_and_QSGD_final_short
    mkdir 3_QSGD_unidirection_max_norm_repeat_num=$repeat_num
    cd 3_QSGD_unidirection_max_norm_repeat_num=$repeat_num
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 35 --save-dir ./result/Signum_and_QSGD_final_short/3_QSGD_unidirection_max_norm_repeat_num=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --enable_max \
    --dist-url tcp://ec2-34-221-38-127.us-west-2.compute.amazonaws.com:1235
done

for repeat_num in 1 2 3 
do
    cd result
    cd Signum_and_QSGD_final_short
    mkdir 3_QSGD_unidirection_l2_norm_repeat_num=$repeat_num
    cd 3_QSGD_unidirection_l2_norm_repeat_num=$repeat_num
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 35 --save-dir ./result/Signum_and_QSGD_final_short/3_QSGD_unidirection_l2_norm_repeat_num=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD \
    --dist-url tcp://ec2-34-221-38-127.us-west-2.compute.amazonaws.com:1235
done


cd imagenet-fast_jiawei
cd imagenet_nv
mkdir result
cd result
mkdir Signum_and_QSGD_final_short
cd ..
for repeat_num in 1 2 3 
do
    cd result
    cd Signum_and_QSGD_final_short
    mkdir 3_Signum_repeat_num=$repeat_num
    cd 3_Signum_repeat_num=$repeat_num
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-3 --seed $repeat_num \
    --epochs 35 --save-dir ./result/Signum_and_QSGD_final_short/3_Signum_repeat_num=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method Signum \
    --dist-url tcp://ec2-34-221-38-127.us-west-2.compute.amazonaws.com:1235
done
'''
'''
#final plot for 3_8_30
#Signum
cd imagenet-fast_jiawei
cd imagenet_nv
mkdir result
cd result
mkdir Signum_and_QSGD_final
cd Signum_and_QSGD_final
mkdir Signum_lr=1e-3
cd Signum_lr=1e-3
mkdir plot
mkdir result_data
cd ..
cd ..
cd ..
ulimit -n 1000000
sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-3 \
--epochs 90 --save-dir ./result/Signum_and_QSGD_final/Signum_lr=1e-3 --world-size 3 --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method Signum \
--dist-url tcp://ec2-34-219-90-70.us-west-2.compute.amazonaws.com:1235

#QSGD with bidirectional
mkdir result
cd result
mkdir Signum_and_QSGD_final
cd Signum_and_QSGD_final
mkdir QSGD_bidirection_lr=1e-2
cd QSGD_bidirection_lr=1e-2
mkdir plot
mkdir result_data
cd ..
cd ..
cd ..
ulimit -n 1000000
sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-2 \
--epochs 90 --save-dir ./result/Signum_and_QSGD_final/QSGD_bidirection_lr=1e-2 --world-size 3 --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method QSGD --bidirection_compress\
--dist-url tcp://ec2-34-219-90-70.us-west-2.compute.amazonaws.com:1235

#QSGD with one-directional
mkdir result
cd result
mkdir Signum_and_QSGD_final
cd Signum_and_QSGD_final
mkdir QSGD_one_direction_lr=1e-1
cd QSGD_one_direction_lr=1e-1
mkdir plot
mkdir result_data
cd ..
cd ..
cd ..
ulimit -n 1000000
sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-1 \
--epochs 90 --save-dir ./result/Signum_and_QSGD_final/QSGD_one_direction_lr=1e-1 --world-size 3 --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method QSGD \
--dist-url tcp://ec2-34-219-90-70.us-west-2.compute.amazonaws.com:1235
'''


'''
#setting for cifar-10
cd imagenet-fast_jiawei
cd imagenet_nv
mkdir result
cd result
mkdir speed_test
cd speed_test
mkdir test_all_gather
cd test_all_gather
mkdir plot
mkdir result_data
cd ..
cd ..
cd ..
ulimit -n 1000000
sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 \
--epochs 40 --save-dir ./result/speed_test/test_all_gather --world-size 3 --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method QSGD --all_gather_commu \
--dist-url tcp://ec2-54-202-114-194.us-west-2.compute.amazonaws.com:1235

'''





'''
#tuning for cifar-10
cd imagenet-fast_jiawei
cd imagenet_nv
mkdir result
cd result
mkdir QSGD_convergence_test_if_bidirection
cd ..
for learning_rate in 0.1 0.01 0.001 0.0001 0.00001 0.000001
do
    cd result
    cd QSGD_convergence_test_if_bidirection
    mkdir 3_QSGD_bidirection_coding_lr=$learning_rate
    cd 3_QSGD_bidirection_coding_lr=$learning_rate
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr $learning_rate \
    --epochs 40 --save-dir ./result/QSGD_convergence_test_if_bidirection/3_QSGD_bidirection_coding_lr=$learning_rate --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --bidirection_compress \
    --dist-url tcp://ec2-52-42-103-144.us-west-2.compute.amazonaws.com:1235
done

for learning_rate in 0.1 0.01 0.001 0.0001 0.00001 0.000001
do
    cd result
    cd QSGD_convergence_test_if_bidirection
    mkdir 3_QSGD_one_direction_coding_lr=$learning_rate
    cd 3_QSGD_one_direction_coding_lr=$learning_rate
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr $learning_rate \
    --epochs 40 --save-dir ./result/QSGD_convergence_test_if_bidirection/3_QSGD_one_direction_coding_lr=$learning_rate --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD \
    --dist-url tcp://ec2-52-42-103-144.us-west-2.compute.amazonaws.com:1235
done
'''


'''
#test standird version
cd imagenet-fast_jiawei
cd imagenet_nv
mkdir result
cd result
mkdir speed_test
cd speed_test
mkdir test
cd test
mkdir plot
mkdir result_data
cd ..
cd ..
cd ..
ulimit -n 1000000
sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 \
--epochs 40 --save-dir ./result/speed_test/test --world-size 3 --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method QSGD \
--dist-url tcp://ec2-52-42-103-144.us-west-2.compute.amazonaws.com:1235
'''