#setting for cifar-10  and disable majority vote
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
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-4 \
--epochs 90 --save-dir ./result/speed_test/test --world-size 3 --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method Signum --disable_majority_vote \
--dist-url tcp://ec2-54-203-186-168.us-west-2.compute.amazonaws.com:1235