#adversary in p3.16x
#example for Signum with random adversary
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
--nproc_per_node=1 --nnodes=3 --node_rank=2 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-3 \
--epochs 90 --save-dir ./result/speed_test/test --world-size 3 --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method Signum \
--enable_adversary --adversary_num 2 --enable_minus_adversary \
--dist-url tcp://ec2-18-237-163-253.us-west-2.compute.amazonaws.com:1235