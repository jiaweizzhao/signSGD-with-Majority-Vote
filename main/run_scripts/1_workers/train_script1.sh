#adversarial test
cd imagenet-fast_jiawei
cd imagenet_nv
mkdir result
cd result
mkdir LARC_test
cd LARC_test
mkdir LARC_default
cd LARC_default
mkdir plot
mkdir result_data
cd ..
cd ..
cd ..
ulimit -n 1000000
sudo /home/ubuntu/anaconda3/envs/pytorch_p36/bin/python3 -m torch.distributed.launch \
--nproc_per_node=4 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.0001 \
--epochs 90 --save-dir ./result/LARC_test/LARC_default --world-size 4 --print-freq 200 \
--dist_backend gloo --all_reduce --signum --warm-up --test_evaluate \
--larc_enable --larc_clip