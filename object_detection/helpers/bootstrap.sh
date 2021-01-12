#!/bin/bash -xe
export PROJECT_ROOT_PATH=${HOME}/object_detection/object_detection
# export MODEL=faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8
# export MODEL=mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8
export MODEL=ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
export PIPELINE_CONFIG_FILE=pipeline_${MODEL}.config

if [[ "$OSTYPE" == "linux-gnu"* ]]; then    
    apt install git wget -y
fi
mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts 

git clone --depth 1 https://github.com/gianrubio/object_detection.git

cd $PROJECT_ROOT_PATH/..
python -m pip install .
cd $PROJECT_ROOT_PATH

ln -s $PROJECT_ROOT_PATH/.. /

wget --quiet http://download.tensorflow.org/models/object_detection/tf2/20200711/$MODEL.tar.gz  
mkdir -p $PROJECT_ROOT_PATH/trainning/models 
tar -xzf $MODEL.tar.gz 
mv $MODEL $PROJECT_ROOT_PATH/trainning/models 
rm $MODEL.tar.gz


rm -rf $PROJECT_ROOT_PATH/trainning/images/test 
rm -rf $PROJECT_ROOT_PATH/trainning/images/train
rm -rf $PROJECT_ROOT_PATH/trainning/assets/
mkdir -p $PROJECT_ROOT_PATH/trainning/assets/

# Partition datase
python $PROJECT_ROOT_PATH/helpers/preprocessing/partition_dataset.py -i $PROJECT_ROOT_PATH/trainning/images -r 0.10 -x
ls $PROJECT_ROOT_PATH/trainning/images/test|wc -l
ls $PROJECT_ROOT_PATH/trainning/images/train|wc -l
rm $PROJECT_ROOT_PATH/trainning/images/*.jpg
rm $PROJECT_ROOT_PATH/trainning/images/*.xml

for type in $(echo "train test")
do
    echo $type
    python ${PROJECT_ROOT_PATH}/helpers/preprocessing/xml_to_csv.py -i ${PROJECT_ROOT_PATH}/trainning/images/$type -o ${PROJECT_ROOT_PATH}/trainning/assets/${type}_labels.csv 
    python ${PROJECT_ROOT_PATH}/helpers/preprocessing/generate_tfrecord.py --image_dir ${PROJECT_ROOT_PATH}/trainning/images/$type --csv_input ${PROJECT_ROOT_PATH}/trainning/assets/${type}_labels.csv  --output_path ${PROJECT_ROOT_PATH}/trainning/assets/${type}.record
done