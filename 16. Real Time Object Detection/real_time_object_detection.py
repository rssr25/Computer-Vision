#import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required= True, help = "path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help = "path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type = float, default=0.2, help = "minimum probability to filter weak detections")

args = vars(ap.parse_args())