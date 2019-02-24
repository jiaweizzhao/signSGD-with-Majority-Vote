#S-3 recover


s3cmd get s3://signum-majority-vote/training_file_history/backup/ --recursive;\
s3cmd get s3://signum-majority-vote/dataset/ILSVRC.tar ~/ILSVRC.tar;\
tar -xvf ILSVRC.tar