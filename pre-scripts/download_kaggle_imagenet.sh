#!/bin/bash

# Download imagnet from Kaggle
source activate fastai
pip install kaggle
chmod 600 /home/ubuntu/.kaggle/kaggle.json #Note: apply your own kaggle.json from kaggle website
kaggle competitions download -c imagenet-object-localization-challenge
tar -xvzf ~/.kaggle/competitions/imagenet-object-localization-challenge/imagenet_object_localization.tar.gz -C ~/



# Save tar to EFS(Optional)
#sudo mv ~/.kaggle/competitions/imagenet-object-localization-challenge/imagenet_object_localization.tar.gz ~/efs_mount