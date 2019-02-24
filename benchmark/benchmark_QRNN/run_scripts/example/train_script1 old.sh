#fix 
cd awd-lstm-lm
mkdir WK_warm_up_tuning
cd WK_warm_up_tuning
mkdir 16_Signum_off_momentum_test
cd ..
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
--nproc_per_node=8 --nnodes=3 --node_rank=0 --master_addr="ec2-54-189-180-185.us-west-2.compute.amazonaws.com" \
--master_port=1235 main_signum.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 60 \
--optimizer adam --lr 1e-4 --momentum 0.9 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
--world-size 24 \
--save-dir ./WK_warm_up_tuning/16_Signum_off_momentum_test/ --distributed 
#--momentun_warm_up



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