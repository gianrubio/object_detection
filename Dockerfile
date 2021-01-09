FROM  tensorflow/tensorflow:devel-gpu
COPY object_detection/packages/tf2/setup.py .
RUN python -m pip install .
COPY . /tensorflow
WORKDIR /tensorflow
CMD ["python", "/tensorflow/object_detection/model_main_tf2.py", "—logtostderr", "--model_dir", "/tensorflow/object_detection/trainning/", "--pipeline_config_path", "/tensorflow/object_detection/trainning/pipeline.config"]