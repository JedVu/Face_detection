""" References:
Origin implementation: https://github.com/sthanhng/yoloface/blob/master/yoloface.py
Blob from images: https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
OpenCV-DNN: https://docs.opencv.org/3.4/d6/d0f/group__dnn.html
Net Object: https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html
Working with Video: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
"""

import argparse
import cv2
import numpy as np
import os
import pickle
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--capture', type=bool, default=False,
                    help='Save capture detection or not')
parser.add_argument('--name', type=str, default=None,
                    help='Target name to save')
args = parser.parse_args()

MODEL = './yolo/yolov3-face.cfg'
WEIGHT = './yolo/yolov3-wider_16000.weights'

DATA_FOLDER = './model/data'
CLASSIFIER = './model/knn.pkl'
LABELS = './model/labels.pkl'

IMG_WIDTH, IMG_HEIGHT = 416, 416
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Load YOLO
net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load Model
_file = open(CLASSIFIER, "rb")
model = pickle.load(_file)

_file = open(LABELS, "rb")
labels = pickle.load(_file)

def _main():
    wind_name = 'Face Detection with YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        
        # Load image to net
        blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT),
                                    [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        output_layers = net.getUnconnectedOutLayersNames()
        outs = net.forward(output_layers)

        # Remove the bounding boxes with low confidence
        boxes = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

        # Run capture mode
        if args.capture == True:
            capture(frame, args.name, boxes)
        
        # Display prediction if there are people
        if len(boxes) != 0: 
            visualize(frame, boxes, model, labels)

        # Display the resulting frame
        cv2.imshow(wind_name, frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    _main()

