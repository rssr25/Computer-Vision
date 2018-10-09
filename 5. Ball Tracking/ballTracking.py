from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2 as cv
import imutils
import time

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default = 64, help = "max buffer size")
args = vars(ap.parse_args())

