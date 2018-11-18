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
--nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.1 \
--epochs 40 --save-dir ./result/speed_test/test_all_gather --world-size 3 --print-freq 50 \
--extra_epochs 0 --compress --signum --communication_method QSGD --all_gather_commu \
--dist-url tcp://ec2-54-203-186-168.us-west-2.compute.amazonaws.com:1235