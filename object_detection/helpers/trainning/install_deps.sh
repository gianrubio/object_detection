#!/bin/bash -xe
export PROJECT_ROOT_PATH=${HOME}/object_detection/object_detection
# export MODEL=faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8
# export MODEL=mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8
export MODEL=ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
export PIPELINE_CONFIG_FILE=pipeline_${MODEL}.config

ln -fs /usr/share/zoneinfo/America/Sao_Paulo /etc/localtime
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata git wget vim python-opencv -y

# pip install awscli 

mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts 

git clone --depth 1 https://github.com/gianrubio/object_detection.git

cd $PROJECT_ROOT_PATH/..
cp $PROJECT_ROOT_PATH/setup.py .
python3 -m pip install .

wget --quiet http://download.tensorflow.org/models/object_detection/tf2/20200711/$MODEL.tar.gz  
mkdir -p $PROJECT_ROOT_PATH/pre-trained-models
mkdir -p $PROJECT_ROOT_PATH/models/my_$MODEL
cp $PROJECT_ROOT_PATH/trainning/$PIPELINE_CONFIG_FILE $PROJECT_ROOT_PATH/models/my_$MODEL
tar -xzf $MODEL.tar.gz 
mv $MODEL $PROJECT_ROOT_PATH/pre-trained-models
rm $MODEL.tar.gz

cd $PROJECT_ROOT_PATH/trainning/
wget http://detectssss.s3-website-eu-west-1.amazonaws.com/train.record 
wget http://detectssss.s3-website-eu-west-1.amazonaws.com/test.record