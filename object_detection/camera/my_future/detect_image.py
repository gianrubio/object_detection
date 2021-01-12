# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:04:32 2020

@author: zhou-
"""

from cv2 import dnn
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
from keras import backend as K
from keras.models import load_model
#from tensorflow_serving.session_bundle import exporter
from keras.models import model_from_config
from keras.models import Sequential,Model
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os

print(tf.__version__)
print(cv2.__version__)
#%% opencv处理适合模型输入的图片
img_file = r"C:\Users\zhou-\Pictures\cat.jpg"
img_cv2 = cv2.imread(img_file)
print("[INFO]Image shape: ", img_cv2.shape)

# 主要图片尺寸要和模型输入匹配（mobilenet要求输入的尺寸为224*224）
inWidth = 224
inHeight = 224
blob = cv2.dnn.blobFromImage(img_cv2,
                                scalefactor=1.0 / 255,
                                size=(inWidth, inHeight),
                                mean=(0, 0, 0),
                                swapRB=False,
                                crop=False)
# blob = np.transpose(blob, (0,2,3,1)) # 适合keras mobilenet网络输入格式
print("[INFO]img shape: ", blob.shape)

#%% 保存keras模型为SaveModel会报错，相关issue见：
# https://github.com/opencv/opencv/issues/16582
model = tf.keras.applications.mobilenet.MobileNet(weights=None)
# model.save('my_model', save_format='tf') # Save model to SavedModel format

# 参考https://github.com/leimao/Frozen_Graph_TensorFlow/blob/master/TensorFlow_v2/train.py

# Save model to SavedModel format
tf.saved_model.save(model, "./models")

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=False)

net = dnn.readNetFromTensorflow('frozen_models/frozen_graph.pb')

# Run a model
net.setInput(blob)
out = net.forward()

# Get a class with a highest score.
out = out.flatten()
classId = np.argmax(out)
confidence = out[classId]

# Put efficiency information.
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
cv2.putText(img_cv2, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

# Print predicted class.
def load_imagenet_classes(file_path):
    '''
    imagenet对应的标签数据如下所示：
    0: 'tench, Tinca tinca',
    1: 'goldfish, Carassius auratus',
    ...
    '''
    classes = []
    contents = None
    with open(file_path,'r') as f:
        contents = f.readlines()
    for cnt in contents:
        cnt = cnt.strip()
        classes.append(cnt.split(':')[1].strip().replace(',',''))
    
    return classes
        
classes = load_imagenet_classes('imagenet_classes.txt')

label = '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
cv2.putText(img_cv2, label, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

cv2.imwrite('output-{}.png'.format(img_file.split('\\')[-1][:-4]), img_cv2)