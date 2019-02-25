#!/bin/bash
#Note: The script is only for AWS instance (Feb, 2019)

#get data from S3 bucket
sudo apt-get install s3cmd

s3cmd --configure  #You need to configure your own S3

s3cmd get s3://signum-majority-vote/training_file_history/imagenet-fast_jiawei --recursive

s3cmd get s3://signum-majority-vote/dataset/ILSVRC.tar

tar -xvf ILSVRC.tar

rm ILSVRC.tar

#install package on pytorch_p36 conda enviroment

source activate pytorch_p36

pip install tensorboardX

#current pytorch verison needs following scripts to run TensorboardX
pip install dataclasses

pip install nvidia-ml-py3

pip install -U protobuf
#

