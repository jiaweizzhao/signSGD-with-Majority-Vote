cd awd-lstm-lm
mkdir WK_3_8_30_final
cd WK_3_8_30_final
mkdir Adam_3_8_30_lr=1e-3
cd ..
/home/ubuntu/anaconda3/envs/qrnn/bin/python3 -u -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=1 --master_addr="0.0.0.0" \
--master_port=1235 main_signum.py --epochs 12 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 \
--dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 240 \
--optimizer adam --lr 1e-3 --momentum 0.9 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN \
--world-size 3 --dist-url tcp://ec2-18-237-137-159.us-west-2.compute.amazonaws.com:1235 \
--save-dir ./WK_3_8_30_final/Adam_3_8_30_lr=1e-3/ --distributed --multi_gpu
