#install env script
conda create -n qrnn python=3.6
source activate qrnn
pip install cupy pynvrtc git+https://github.com/PermiJW/pytorch-qrnn
conda install pytorch torchvision -c pytorch
pip install tensorboardX
pip install cython

#download dataset
cd awd...blabla
sh getdata.sh

#32 machines
cd imagenet-fast_jiawei
cd imagenet_nv
mkdir result
cd result
mkdir Sep_10_new_tuning_spilt_dataset
cd Sep_10_new_tuning_spilt_dataset
#mkdir num=7_Signum_test_small_bz_64
mkdir num=7_Signum_new_WD_0.000001
cd num=7_Signum_new_WD_0.000001
mkdir plot
mkdir result_data
cd ..
cd ..
cd ..
ulimit -n 1000000

adam 1e-3
num 3_Signum_lr_1e-4_decay_early
#new1
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 main_signum.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 60 \
--optimizer signum --lr 1e-4 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 4 --model QRNN \
--world-size 3 --dist-url tcp://ec2-34-222-72-158.us-west-2.compute.amazonaws.com:1235 \
--save-dir ./result/3_Signum_lr_1e-4_decay_early/ --distributed

#2 workers
#new2
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 main_signum.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 60 \
--optimizer signum --lr 1e-4 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 4 --model QRNN \
--world-size 2 --dist-url tcp://ec2-54-245-131-183.us-west-2.compute.amazonaws.com:1235 \
--save-dir ./result/3_Signum_lr_1e-4_decay_early/ --distributed
#new3
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="0.0.0.0" \
--master_port=1235 main_signum.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 60 \
--optimizer signum --lr 1e-4 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 4 --model QRNN \
--world-size 2 --dist-url tcp://ec2-54-245-131-183.us-west-2.compute.amazonaws.com:1235 \
--save-dir ./result/3_Signum_lr_1e-4_decay_early/ --distributed

#baseline
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u main_signum.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 60 \
--optimizer adam --lr 1e-3 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
--world-size 3 --dist-url tcp://ec2-34-222-72-158.us-west-2.compute.amazonaws.com:1235 \
--save-dir ./result/1_Adam_baseline/ 



#For LSTM
sudo /home/ubuntu/anaconda3/envs/qrnn/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr="0.0.0.0" \
--master_port=1235 main_signum.py --epochs 500 --nlayers 3 --emsize 200 --nhid 1000 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.25 --dropouti 0.1 --dropout 0.1 --wdrop 0.5 --wdecay 1.2e-6 --bptt 150 --batch_size 128 \
--optimizer adam --lr 2e-3 --data data/pennchar --save PTBC.pt --when 300 400 \
--world-size 2 --dist-url tcp://ec2-34-222-72-158.us-west-2.compute.amazonaws.com:1235



sudo /home/ubuntu/anaconda3/envs/qrnn/bin/python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=7 --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.0001 \
--epochs 90 --save-dir ./result/Sep_10_new_tuning_spilt_dataset/num=7_Signum_new_WD_0.000001 --world-size 7 --print-freq 200 \
--extra_epochs 0 \
--dist-url tcp://ec2-34-222-72-158.us-west-2.compute.amazonaws.com:1235

python -u main.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 60 --optimizer adam --lr 1e-3 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN





