#!/usr/bin/env python

import paramiko


server_list = {
#p3.16x
#'1':'ec2-54-185-254-143.us-west-2.compute.amazonaws.com'
#'2':'ec2-18-237-47-220.us-west-2.compute.amazonaws.com'
#'3':'ec2-35-165-31-92.us-west-2.compute.amazonaws.com'
#group1 + addtional
'i-05a8d5e118d1a391a': 'ec2-35-167-102-54.us-west-2.compute.amazonaws.com',\
'i-05fcb6eeafc2ae3f4': 'ec2-34-209-62-141.us-west-2.compute.amazonaws.com',\
'i-0982a6284ea8feaa8': 'ec2-54-245-59-196.us-west-2.compute.amazonaws.com',\
'i-0f7ed93ad03cbd6e0': 'ec2-34-209-238-130.us-west-2.compute.amazonaws.com',\
'i-05a64c634aaacf690': 'ec2-34-213-170-68.us-west-2.compute.amazonaws.com',\
'i-0709a3f272f613c99': 'ec2-35-167-247-55.us-west-2.compute.amazonaws.com',\
'i-09c79181de2153b57': 'ec2-54-202-238-48.us-west-2.compute.amazonaws.com'
#'i-0f05a98dfcb5de301':'ec2-34-220-194-215.us-west-2.compute.amazonaws.com',\
#'i-0f527dbf350eeabbd':'ec2-35-165-67-21.us-west-2.compute.amazonaws.com',\
#'fast_ai_9':'ec2-34-221-191-227.us-west-2.compute.amazonaws.com',\
#'fast_ai_10':'ec2-54-245-131-183.us-west-2.compute.amazonaws.com',\
#'fast_ai_11':'ec2-54-190-77-19.us-west-2.compute.amazonaws.com',\
#'fast_ai_12':'ec2-18-237-207-139.us-west-2.compute.amazonaws.com',\
#'fast_ai_13':'ec2-34-209-28-245.us-west-2.compute.amazonaws.com',\
#'fast_ai_14':'ec2-52-89-128-156.us-west-2.compute.amazonaws.com',\
#'fast_ai_15':'ec2-52-42-166-71.us-west-2.compute.amazonaws.com',\
#'fast_ai_16':'ec2-34-214-22-27.us-west-2.compute.amazonaws.com',\
#'fast_ai_17':'ec2-35-165-112-126.us-west-2.compute.amazonaws.com'\
#group 4
#'i-0094ff4f64f218e2b':'ec2-34-221-224-37.us-west-2.compute.amazonaws.com',\
#'i-07de70d694f9a54ed':'ec2-54-190-111-181.us-west-2.compute.amazonaws.com',\
#'i-05dd84dc8c023df5f':'ec2-35-164-12-182.us-west-2.compute.amazonaws.com',\
#'i-05b0c3a1b07cf0b9a':'ec2-54-191-213-147.us-west-2.compute.amazonaws.com',\
#'i-058e1fd87cbe8a542':'ec2-54-184-164-145.us-west-2.compute.amazonaws.com',\
#'i-04cf82bd3fc4feb5e':'ec2-35-162-34-124.us-west-2.compute.amazonaws.com',\
#'i-04226fefc80e8989a':'ec2-54-149-242-139.us-west-2.compute.amazonaws.com',\
#group 6 1
#'i-08ec3104e3dcaa550':'ec2-54-186-47-244.us-west-2.compute.amazonaws.com',\
#'i-0997972abcf79536c':'ec2-54-200-200-211.us-west-2.compute.amazonaws.com',\
#'i-0b44f4bad9eab0363':'ec2-54-185-100-51.us-west-2.compute.amazonaws.com',\
#group 6 2
#'i-0b817e762b9d8a8bf':'ec2-35-163-44-99.us-west-2.compute.amazonaws.com',\
#'i-0c364e72c549c5eb5':'ec2-34-221-251-62.us-west-2.compute.amazonaws.com',\
#'i-0c591137b069063aa':'ec2-34-221-244-28.us-west-2.compute.amazonaws.com',\
#'i-0c762f11e7a373136':'ec2-54-203-17-94.us-west-2.compute.amazonaws.com'
#additional instance
#'additional':'ec2-34-216-66-202.us-west-2.compute.amazonaws.com'
#group 2
#'i-0f2b61889acfcd5b1':'ec2-34-217-119-248.us-west-2.compute.amazonaws.com'#add to group1
#'i-0a94b85d9cc93606c':'ec2-18-237-227-229.us-west-2.compute.amazonaws.com',\
#'i-091da8cc042b00136':'ec2-34-219-105-255.us-west-2.compute.amazonaws.com',\
#'i-0732afe9141942cbf':'ec2-18-236-219-41.us-west-2.compute.amazonaws.com',\
#'i-06ce99ed3a8067e06':'ec2-34-210-122-236.us-west-2.compute.amazonaws.com',\
#'i-064ec77b87e79944b':'ec2-34-222-68-78.us-west-2.compute.amazonaws.com',\
#'i-04fcf869400b723e6':'ec2-18-237-66-248.us-west-2.compute.amazonaws.com'
}

def start_sever():
    for index, sever in enumerate(server_list):

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server_list[sever], 22, 'ubuntu',
        key_filename='/Users/jonah/.ssh/megadata-OR.pem')

        sftp = paramiko.SFTPClient.from_transport(ssh.get_transport())
        #.sh file upload
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_Krum/run_scripts/7_example/train_script' + str(index + 1) + '.sh',\
            '/home/ubuntu/train_script.sh')

        #benchmark_main.py upload
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_Krum/benchmark_main.py',\
            '/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/benchmark_main.py')
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_Krum/Imagefolder_train_val.py',\
            '/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/Imagefolder_train_val.py')
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_Krum/resnet.py',\
            '/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/resnet.py')
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_Krum/QSGD_gpu.py',\
            '/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/QSGD_gpu.py')
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_Krum/QSGD_optimizer.py',\
            '/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/QSGD_optimizer.py')
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_Krum/byzantine_utils.py',\
            '/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/byzantine_utils.py')
        #distributed_model upload
        #sftp.put('/Users/jonah/Desktop/signum/source code/imagenet-fast_jiawei/imagenet_nv/distributed_model.py',\
            #'/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/distributed_model.py')
        #Signum_SGD.py
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_Krum/Signum_optimizer.py',\
            '/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/Signum_optimizer.py')
        #compressor.py
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_Krum/compressor.py',\
            '/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/compressor.py')

        '''
        #extend file for ATOMO
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/modified_version_for_ATOMO/codings/__init__.py',\
            '/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/codings/__init__.py')
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/modified_version_for_ATOMO/codings/coding.py',\
            '/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/codings/coding.py')
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/modified_version_for_ATOMO/codings/svd.py',\
            '/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/codings/svd.py')
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/modified_version_for_ATOMO/codings/utils.py',\
            '/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/codings/utils.py')
        '''

        #Imagefolder_train_val
        #sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/modified_version_for_ATOMO/old_version_code/Imagefolder_train_val.py',\
            #'/home/ubuntu/imagenet-fast_jiawei/imagenet_nv/Imagefolder_train_val.py')

        #just for test
        
        if index == 0:
            continue
        
        
        
        #first time screen need to use -S, then -r 
        ssh.exec_command('sudo chmod -R 777 /home/ubuntu/train_script.sh;screen -s /home/ubuntu/train_script.sh -L -dmS test')


if __name__ == "__main__":
    start_sever()
        



