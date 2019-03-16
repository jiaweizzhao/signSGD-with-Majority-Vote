#adversarial test
cd imagenet-fast_jiawei
cd imagenet_nv
for LARC_trust_co in 1e-2
do
    mkdir result
    cd result
    mkdir LARC_tuning=short_epoch
    cd LARC_tuning=short_epoch
    mkdir LARC_trust_coefficient=$LARC_trust_co=with_clip=added_WD
    cd LARC_trust_coefficient=$LARC_trust_co=with_clip=added_WD
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/pytorch_p36/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=4 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 1e-4 \
    --epochs 45 --save-dir ./result/LARC_tuning=short_epoch/LARC_trust_coefficient=$LARC_trust_co=with_clip=added_WD --world-size 4 --print-freq 200 \
    --dist_backend gloo --all_reduce --signum --warm-up --test_evaluate \
    --larc_enable --larc_clip --larc_trust_coefficient $LARC_trust_co \
    --decay-int 15 --weight-decay 1e-4 #extra setting
done




'''
cd imagenet-fast_jiawei
cd imagenet_nv
for LARC_trust_co in 1e-2 1e-3 1e-4 1e-5
do
    mkdir result
    cd result
    mkdir LARC_tuning=short_epoch
    cd LARC_tuning=short_epoch
    mkdir LARC_trust_coefficient=$LARC_trust_co
    cd LARC_trust_coefficient=$LARC_trust_co
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    sudo /home/ubuntu/anaconda3/envs/pytorch_p36/bin/python3 -m torch.distributed.launch \
    --nproc_per_node=4 benchmark_main.py ~/ILSVRC/Data/CLS-LOC -a resnet50 -b 128 --lr 0.0001 \
    --epochs 90 --save-dir ./result/LARC_tuning=short_epoch/LARC_trust_coefficient=$LARC_trust_co --world-size 4 --print-freq 200 \
    --dist_backend gloo --all_reduce --signum --warm-up --test_evaluate \
    --larc_enable --larc_clip --larc_trust_coefficient $LARC_trust_co \
    --decay-int 15 #extra setting
done
'''

