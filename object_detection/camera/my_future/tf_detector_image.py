#!/usr/bin/env python
# coding: utf-8
"""
Object Detection (On Video) From TF2 Saved Model
=====================================
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse

tf.get_logger().setLevel("ERROR")  # Suppress TensorFlow logging (2)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    help="Folder that the Saved Model is Located In",
    # default="/Users/grubio/.keras/datasets/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/",
    default="/Users/grubio/Downloads/image/multiple-image-detection/models/research/object_detection/trainning/",
)
parser.add_argument(
    "--labels",
    help="Where the Labelmap is Located",
    default="/Users/grubio/Downloads/image/multiple-image-detection/models/research/object_detection/trainning/13-jan/exported-model-12-jan/saved_model/saved_model.pbtxt",
)
parser.add_argument(
    "--video",
    help="Name of the video to perform detection on. To run detection on multiple images, use --imagedir",
    default="/Users/grubio/Downloads/image/camera/my_future/petra_tonica_maguary.mp4",
)
parser.add_argument(
    "--threshold",
    help="Minimum confidence threshold for displaying detected objects",
    default=0.5,
)

args = parser.parse_args()
# Enable GPU dynamic memory allocation
# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO IMAGE DIRECTORY
VIDEO_PATHS = args.video


# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = args.labels

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(args.threshold)

# Load the model
# ~~~~~~~~~~~~~~
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print("Loading model...", end="")
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)


end_time = time.time()
elapsed_time = end_time - start_time
print("Done! Took {} seconds".format(elapsed_time))

# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True
)

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # Suppress Matplotlib warnings


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

IMAGE_PATHS = ["/Users/grubio/Downloads/IMG_1061.jpg","/Users/grubio/Downloads/IMG_1062.jpg","/Users/grubio/Downloads/IMG_1063.jpg","/Users/grubio/Downloads/IMG_1064.jpg"]


for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.show()
    plt.savefig((f"{image_path}.jpg"))

    print('Done')
plt.show()

# sphinx_gallery_thumbnail_number = 2