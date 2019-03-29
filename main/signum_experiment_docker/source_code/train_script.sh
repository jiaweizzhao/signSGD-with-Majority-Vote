#adversarial test 
for LARC_trust_co in 1e-3
do
    mkdir result
    cd result
    mkdir LARC_1
    cd LARC_1
    mkdir with_co=$LARC_trust_co
    cd with_co=$LARC_trust_co
    mkdir plot
    mkdir result_data
    cd ..
    cd ..
    cd ..
    ulimit -n 1000000
    python -m torch.distributed.launch \
    --nproc_per_node=8 benchmark_main.py /app/dataset_mirror -a resnet50 -b 128 --lr 1e-4 \
    --epochs 45 --save-dir ./result/LARC_1/with_co=$LARC_trust_co --world-size 8 --print-freq 200 -j 8 \
    --dist_backend gloo --all_reduce --signum --warm-up --test_evaluate \
    --larc_enable --larc_clip --larc_trust_coefficient $LARC_trust_co \
    --cpp_extend_load \
    --decay-int 15 --weight-decay 1e-4 #extra setting


done


echo "All experiments have finished! :)"
