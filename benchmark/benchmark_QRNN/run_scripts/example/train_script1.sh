cd awd-lstm-lm
mkdir WK_1_8_batch_size_explore_final
for batch_size in 60
do
    cd WK_1_8_batch_size_explore_final
    mkdir Signum_bz=$batch_size=lr=1e-4_multi_batch_size
    cd ..
    /home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m main_signum.py --epochs 12 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
    --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size $batch_size \
    --optimizer signum --lr 1e-4 --momentum 0.9 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
    --world-size 1 --multi_batch_size \
    --save-dir ./WK_1_8_batch_size_explore_final/Signum_bz=$batch_size=lr=1e-4_multi_batch_size/ --multi_gpu --single_worker
done


'''
cd awd-lstm-lm
mkdir WK_1_8_batch_size_explore_new
for learning_rate in 1e-3 1e-4 1e-5 #1e-2 finish 4 epoch
do
    cd WK_1_8_batch_size_explore_new
    mkdir Signum_bz=60_lr=$learning_rate
    cd ..
    /home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m main_signum.py --epochs 4 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
    --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 60 \
    --optimizer signum --lr $learning_rate --momentum 0.9 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
    --world-size 1 \
    --save-dir ./WK_1_8_batch_size_explore_new/Signum_bz=60_lr=$learning_rate/ --multi_gpu --single_worker
done

cd awd-lstm-lm
mkdir WK_1_8_batch_size_explore_new
for learning_rate in 1e-3 1e-4 1e-5
do
    cd WK_1_8_batch_size_explore_new
    mkdir Signum_bz=120_lr=$learning_rate
    cd ..
    /home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m main_signum.py --epochs 4 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
    --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 120 \
    --optimizer signum --lr $learning_rate --momentum 0.9 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
    --world-size 1 \
    --save-dir ./WK_1_8_batch_size_explore_new/Signum_bz=120_lr=$learning_rate/ --multi_gpu --single_worker
done

cd awd-lstm-lm
mkdir WK_1_8_batch_size_explore_new
for learning_rate in 1e-3 1e-4 1e-5
do
    cd WK_1_8_batch_size_explore_new
    mkdir Signum_bz=180_lr=$learning_rate
    cd ..
    /home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m main_signum.py --epochs 4 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
    --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 180 \
    --optimizer signum --lr $learning_rate --momentum 0.9 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
    --world-size 1 \
    --save-dir ./WK_1_8_batch_size_explore_new/Signum_bz=180_lr=$learning_rate/ --multi_gpu --single_worker
done

cd awd-lstm-lm
mkdir WK_1_8_batch_size_explore_new
for learning_rate in 1e-3 1e-4 1e-5
do
    cd WK_1_8_batch_size_explore_new
    mkdir Signum_bz=240_lr=$learning_rate
    cd ..
    /home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m main_signum.py --epochs 4 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
    --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 240 \
    --optimizer signum --lr $learning_rate --momentum 0.9 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
    --world-size 1 \
    --save-dir ./WK_1_8_batch_size_explore_new/Signum_bz=240_lr=$learning_rate/ --multi_gpu --single_worker
done
'''

'''
cd awd-lstm-lm
mkdir WK_3_8_30_final
cd WK_3_8_30_final
mkdir Adam_3_8_30_lr=1e-3
cd ..
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 main_signum.py --epochs 12 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 240 \
--optimizer adam --lr 1e-3 --momentum 0.9 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
--world-size 3 --dist-url tcp://ec2-18-237-137-159.us-west-2.compute.amazonaws.com:1235 \
--save-dir ./WK_3_8_30_final/Adam_3_8_30_lr=1e-3/ --distributed --multi_gpu
'''








'''
cd awd-lstm-lm
mkdir WK_full_version_lr_tuning_3_8_30
for learning_rate in 1e-4 1e-5
do 
    cd WK_full_version_lr_tuning_3_8_30
    mkdir 3_8_Signum_bz=240_lr=$learning_rate
    cd ..
    /home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 main_signum.py --epochs 10 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
    --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 240 \
    --optimizer signum --lr $learning_rate --momentum 0.9 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
    --world-size 3 --dist-url tcp://ec2-18-237-230-210.us-west-2.compute.amazonaws.com:1235 \
    --save-dir ./WK_full_version_lr_tuning_3_8_30/3_8_Signum_bz=240_lr=$learning_rate/ --distributed --multi_gpu

done

mkdir WK_full_version_momentum_tuning_3_8_30
for momentum in 0.9 0.5 0.99
do 
    cd WK_full_version_momentum_tuning_3_8_30
    mkdir 3_8_Signum_bz=240_momentum=$momentum
    cd ..
    /home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
    --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
    --master_port=1235 main_signum.py --epochs 10 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
    --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 240 \
    --optimizer signum --lr 1e-3 --momentum $momentum --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
    --world-size 3 --dist-url tcp://ec2-18-237-230-210.us-west-2.compute.amazonaws.com:1235 \
    --save-dir ./WK_full_version_momentum_tuning_3_8_30/3_8_Signum_bz=240_momentum=$momentum/ --distributed --multi_gpu

done
'''




'''
cd awd-lstm-lm
mkdir WK_full_version
cd WK_full_version
mkdir 3_8_Signum_final_bz=240_lr_5e-4_to_1e-4
cd ..
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 main_signum.py --epochs 24 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 240 \
--optimizer signum --lr 5e-4 --momentum 0.9 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
--world-size 3 --dist-url tcp://ec2-34-211-38-48.us-west-2.compute.amazonaws.com:1235 \
--save-dir ./WK_full_version/3_8_Signum_final_bz=240_lr_5e-4_to_1e-4/ --distributed --multi_gpu --momentun_warm_up






cd awd-lstm-lm
mkdir WK_full_version
cd WK_full_version
mkdir 3_8_Adam_final_bz=240
cd ..
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 main_signum.py --epochs 12 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 240 \
--optimizer adam --lr 1e-3 --momentum 0.9 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
--world-size 3 --dist-url tcp://ec2-34-211-38-48.us-west-2.compute.amazonaws.com:1235 \
--save-dir ./WK_full_version/3_8_Adam_final_bz=240/ --distributed --multi_gpu
#--multi_gpu

'''


'''
#fix 
cd awd-lstm-lm
mkdir WK_full_version
for momentum in 0.9 0.5 0.99 0
do
    echo $learning_rate
    for learning_rate in 1e-2 1e-3 1e-4 
    do
        echo lr=$learning_rate=momentum=$momentum
        cd WK_full_version
        mkdir 3_Signum_full_lr=1e-3=momentum=0.5_0.9
        cd ..
        /home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
        --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
        --master_port=1235 main_signum.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
        --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 360 \
        --optimizer signum --lr $learning_rate --momentum $momentum --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
        --world-size 3 --dist-url tcp://ec2-34-211-38-48.us-west-2.compute.amazonaws.com:1235 \
        --save-dir ./WK_full_version/3_Signum_full_lr=$learning_rate=momentum=$momentum/ --distributed --tuning_mode --multi_gpu
    done
done
'''

'''
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
--nproc_per_node=7 main_signum.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 60 \
--optimizer sgd_distribute --lr 1e-3 --momentum 0.5 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
--world-size 7 --dist-backend nccl \
--save-dir ./WK_final_tuning_nccl/7_Signum_lr=1e-3_0.5_0.9/ --distributed --momentun_warm_up
'''
'''
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u main_signum.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 360 \
--optimizer adam --lr 1e-3 --momentum 0.5 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
--world-size 7 --dist-backend nccl \
--save-dir ./WK_final_tuning_nccl/7_Signum_lr=1e-3_0.5_0.9/ --multi_gpu

'''

'''
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr="0.0.0.0" \
--master_port=1235 main_signum.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 60 \
--optimizer adam --lr 1e-3 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
--world-size 3 --dist-url tcp://ec2-35-162-17-192.us-west-2.compute.amazonaws.com:1235 \
--save-dir ./tunning/3_Adam_lr_base/ --distributed


#one worker signum
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u main_signum.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 60 \
--optimizer signum --lr 1e-4 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
--world-size 3 --dist-url tcp://ec2-35-162-17-192.us-west-2.compute.amazonaws.com:1235 \
--save-dir ./tunning/1_Signum_1e-4/ 
'''