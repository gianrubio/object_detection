#!/bin/bash -xe
export PROJECT_ROOT_PATH=${HOME}/object_detection/object_detection
# export MODEL=faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8
# export MODEL=mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8
export MODEL=ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
export PIPELINE_CONFIG_FILE=pipeline_${MODEL}.config

if [[ "$OSTYPE" == "linux-gnu"* ]]; then    
    apt install git wget vim -y
fi
pip install awscli
mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts 

git clone --depth 1 https://github.com/gianrubio/object_detection.git

cd $PROJECT_ROOT_PATH../
cp $PROJECT_ROOT_PATH/setup.py .
python -m pip install .


wget --quiet http://download.tensorflow.org/models/object_detection/tf2/20200711/$MODEL.tar.gz  
mkdir -p $PROJECT_ROOT_PATH/trainning/models 
tar -xzf $MODEL.tar.gz 
mv $MODEL $PROJECT_ROOT_PATH/trainning/models 
rm $MODEL.tar.gz
