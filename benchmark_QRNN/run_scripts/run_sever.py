#!/usr/bin/env python

import paramiko


server_list = {
#group3
#'i-033e468c44c0f03c8':'ec2-35-162-17-192.us-west-2.compute.amazonaws.com',\
#'i-03eddcf07180b4580':'ec2-34-214-128-110.us-west-2.compute.amazonaws.com',\
#'i-06e76eb7725bd7c66':'ec2-18-237-181-190.us-west-2.compute.amazonaws.com'
#group7
#'i-0827891c35645cd4a':'ec2-34-222-79-76.us-west-2.compute.amazonaws.com',\
#'i-084b5e1da6ade3799':'ec2-34-220-255-0.us-west-2.compute.amazonaws.com',\
#'i-0a7e0dbe23357a2b4':'ec2-52-88-19-12.us-west-2.compute.amazonaws.com'
#group8
#'i-0d7e1f885efdc27a9':'ec2-54-185-71-29.us-west-2.compute.amazonaws.com',\
#'i-0f5803ececeda828b':'ec2-18-237-187-228.us-west-2.compute.amazonaws.com',\
#'i-0e7b4416e2f81af3a':'ec2-34-219-105-89.us-west-2.compute.amazonaws.com'
#p3.16x
#'1':'ec2-54-189-180-185.us-west-2.compute.amazonaws.com',\
#'2':'ec2-18-237-201-40.us-west-2.compute.amazonaws.com',\
#'3':'ec2-54-149-119-211.us-west-2.compute.amazonaws.com'
'5':'ec2-54-212-161-111.us-west-2.compute.amazonaws.com',\
#'6':'ec2-18-236-219-174.us-west-2.compute.amazonaws.com',\
#'7':'ec2-54-149-187-90.us-west-2.compute.amazonaws.com'

}

def start_sever():
    for index, sever in enumerate(server_list):

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server_list[sever], 22, 'ubuntu',
        key_filename='/Users/jonah/.ssh/megadata-OR.pem')

        sftp = paramiko.SFTPClient.from_transport(ssh.get_transport())
        #.sh file upload
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_QRNN/run_scripts/example/train_script' + str(index + 1) + '.sh',\
            '/home/ubuntu/train_script.sh')
        
        #Signum_SGD.py upload
        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_QRNN/Signum_SGD.py',\
            '/home/ubuntu/awd-lstm-lm/Signum_SGD.py')

        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_QRNN/utils.py',\
            '/home/ubuntu/awd-lstm-lm/utils.py')

        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_QRNN/main_signum.py',\
            '/home/ubuntu/awd-lstm-lm/main_signum.py')

        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_QRNN/splitcross.py',\
            '/home/ubuntu/awd-lstm-lm/splitcross.py')

        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_QRNN/model.py',\
            '/home/ubuntu/awd-lstm-lm/model.py')

        sftp.put('/Users/jonah/Desktop/signum/source code/signSGD-with-Majority-Vote/benchmark_QRNN/compressor.py',\
            '/home/ubuntu/awd-lstm-lm/compressor.py')

        #just for test
        '''
        if index == 0:
            continue
        '''
        
        
        #first time screen need to use -S, then -r 
        #ssh.exec_command('chmod -R 777 /home/ubuntu/train_script.sh;screen -s /home/ubuntu/train_script.sh -L -dmS test')


if __name__ == "__main__":
    start_sever()
        



