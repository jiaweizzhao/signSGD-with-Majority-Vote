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
    --nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr="0.0.0.0" \
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
    --nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr="0.0.0.0" \
    --master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr $learning_rate \
    --epochs 40 --save-dir ./result/QSGD_convergence_test_if_bidirection/3_QSGD_one_direction_coding_lr=$learning_rate --world-size 3 --print-freq 50 \
    --extra_epochs 0 --compress --signum --communication_method QSGD \
    --dist-url tcp://ec2-52-42-103-144.us-west-2.compute.amazonaws.com:1235
done



'''
#setting for cifar-10
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
--nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 \
--epochs 40 --save-dir ./result/speed_test/test --world-size 3 --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method QSGD \
--dist-url tcp://ec2-52-42-103-144.us-west-2.compute.amazonaws.com:1235
'''