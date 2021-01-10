!#/bin/bash -xe
PROJECT_ROOT_PATH=$(pwd)/object_detection
MODEL=faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8
PIPELINE_CONFIG_FILE=pipeline_${MODEL}.config

if [[ "$OSTYPE" == "linux-gnu"* ]]; then    
    apt install git wget -y
fi

ssh-keyscan github.com >> ~/.ssh/known_hosts 
git clone https://github.com/gianrubio/object_detection.git

cd $PROJECT_ROOT_PATH 

python -m pip install .

wget --quiet http://download.tensorflow.org/models/object_detection/tf2/20200711/$MODEL.tar.gz  
mkdir -p $PROJECT_ROOT_PATH/trainning/models 
tar -xzf $MODEL.tar.gz && mv $MODEL $PROJECT_ROOT_PATH/trainning/models 
rm $MODEL.tar.gz


rm -rf $PROJECT_ROOT_PATH/trainning/images/test 
rm -rf $PROJECT_ROOT_PATH/trainning/images/train
rm -rf $PROJECT_ROOT_PATH/trainning/assets/
mkdir -p $PROJECT_ROOT_PATH/trainning/assets/

# Partition datase
python $PROJECT_ROOT_PATH/helpers/preprocessing/partition_dataset.py -i $PROJECT_ROOT_PATH/trainning/images -r 0.15 -x
for type in "train test"
do
    python $PROJECT_ROOT_PATH/helpers/preprocessing/xml_to_csv.py -i $PROJECT_ROOT_PATH/trainning/images/$type -o $PROJECT_ROOT_PATH/trainning/assets/${type}_labels.csv
    python $PROJECT_ROOT_PATH/helpers/preprocessing/generate_tfrecord.py --image_dir=$PROJECT_ROOT_PATH/trainning/images/$type --csv_input={PROJECT_ROOT_PATH}/trainning/assets/${type}_labels.csv  --output_path=$PROJECT_ROOT_PATH/trainning/assets/${type}.record
done