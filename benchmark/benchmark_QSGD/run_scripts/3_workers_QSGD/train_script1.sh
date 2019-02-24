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
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 90 --save-dir ./result/QSGD_final_multi_level_test/QSGD_normal_norm=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --qsgd_level 2 --enable_max --all_reduce \
    --dist-url tcp://ec2-52-11-11-115.us-west-2.compute.amazonaws.com:1235
done



'''
for repeat_num in 1 2 3 
do
    cd result
    cd QSGD_final_multi_level
    mkdir QSGD_2bit_max_norm=$repeat_num
    cd QSGD_2bit_max_norm=$repeat_num
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 90 --save-dir ./result/QSGD_final_multi_level/QSGD_2bit_max_norm=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --qsgd_level 2 --enable_max --all_reduce \
    --dist-url tcp://ec2-54-202-115-191.us-west-2.compute.amazonaws.com:1235
done

for repeat_num in 1 2 3 
do
    cd result
    cd QSGD_final_multi_level
    mkdir QSGD_4bit_max_norm=$repeat_num
    cd QSGD_4bit_max_norm=$repeat_num
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 90 --save-dir ./result/QSGD_final_multi_level/QSGD_4bit_max_norm=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --qsgd_level 4 --enable_max --all_reduce \
    --dist-url tcp://ec2-54-202-115-191.us-west-2.compute.amazonaws.com:1235

done

for repeat_num in 1 2 3 
do
    cd result
    cd QSGD_final_multi_level
    mkdir QSGD_8bit_max_norm=$repeat_num
    cd QSGD_8bit_max_norm=$repeat_num
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 90 --save-dir ./result/QSGD_final_multi_level/QSGD_8bit_max_norm=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --qsgd_level 8 --enable_max --all_reduce \
    --dist-url tcp://ec2-54-202-115-191.us-west-2.compute.amazonaws.com:1235

done

'''


'''
for repeat_num in 1 2 3 
do
    cd result
    cd QSGD_final_multi_level
    mkdir QSGD_2bit_max_norm=$repeat_num
    cd QSGD_2bit_max_norm=$repeat_num
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 90 --save-dir ./result/QSGD_final_multi_level/QSGD_2bit_max_norm=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --qsgd_level 2 --enable_max --all_reduce \
    --dist-url tcp://ec2-54-202-115-191.us-west-2.compute.amazonaws.com:1235
done

for repeat_num in 1 2 3 
do
    cd result
    cd QSGD_final_multi_level
    mkdir QSGD_4bit_max_norm=$repeat_num
    cd QSGD_4bit_max_norm=$repeat_num
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 90 --save-dir ./result/QSGD_final_multi_level/QSGD_4bit_max_norm=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --qsgd_level 4 --enable_max --all_reduce \
    --dist-url tcp://ec2-54-202-115-191.us-west-2.compute.amazonaws.com:1235

done

for repeat_num in 1 2 3 
do
    cd result
    cd QSGD_final_multi_level
    mkdir QSGD_8bit_max_norm=$repeat_num
    cd QSGD_8bit_max_norm=$repeat_num
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 --seed $repeat_num \
    --epochs 90 --save-dir ./result/QSGD_final_multi_level/QSGD_8bit_max_norm=$repeat_num --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --qsgd_level 8 --enable_max --all_reduce \
    --dist-url tcp://ec2-54-202-115-191.us-west-2.compute.amazonaws.com:1235

done
'''


'''
#tuning for cifar-10 and 2-bit QSGD
cd imagenet-fast_jiawei
cd imagenet_nv
mkdir result
cd result
mkdir QSGD_tuning_multi_level
cd ..
for learning_rate in 0.1 0.01 0.001 0.0001 
do
    cd result
    cd QSGD_tuning_multi_level
    mkdir QSGD_2bit_lr=$learning_rate
    cd QSGD_2bit_lr=$learning_rate
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr $learning_rate \
    --epochs 40 --save-dir ./result/QSGD_tuning_multi_level/QSGD_2bit_lr=$learning_rate --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --qsgd_level 2 \
    --dist-url tcp://ec2-54-202-115-191.us-west-2.compute.amazonaws.com:1235
done

for learning_rate in 0.1 0.01 0.001 0.0001 
do
    cd result
    cd QSGD_tuning_multi_level
    mkdir QSGD_4bit_lr=$learning_rate
    cd QSGD_4bit_lr=$learning_rate
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr $learning_rate \
    --epochs 40 --save-dir ./result/QSGD_tuning_multi_level/QSGD_4bit_lr=$learning_rate --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --qsgd_level 4 \
    --dist-url tcp://ec2-54-202-115-191.us-west-2.compute.amazonaws.com:1235
done

for learning_rate in 0.1 0.01 0.001 0.0001 
do
    cd result
    cd QSGD_tuning_multi_level
    mkdir QSGD_8bit_lr=$learning_rate
    cd QSGD_8bit_lr=$learning_rate
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/fastai/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr $learning_rate \
    --epochs 40 --save-dir ./result/QSGD_tuning_multi_level/QSGD_8bit_lr=$learning_rate --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD --qsgd_level 8 \
    --dist-url tcp://ec2-54-202-115-191.us-west-2.compute.amazonaws.com:1235
done
'''



'''
#repeat experiment
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
