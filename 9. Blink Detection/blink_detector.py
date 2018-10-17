#we know that in facial landmarking we have different indices of the points that we get for the landmarks.
#we can extract only the points that the eyes are surrounded by and then find out the EAR (Eye Aspect Ratio)
#to determine if the blink happened or not. 

#there are 6 xy coordinates for the eye. Starting at the left corner of the eye, (right of the person) and then 
#going clockwise. There is a relation between the width and height of these coordinates.


#importing the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
	#compute the euclidean distances between the two sets of 
	#vertical eye landmarks (x,y) -coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	#compute the euclidean distance between the horizontal eye landmark
	#(x,y) coordinates
	C = dist.euclidean(eye[0], eye[3])

	#compute the eye aspect ratio
	ear = (A + B)/ (2.0 * C)

	#return the eye aspect ratio
	return ear


#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-detector", required = True, help = "path to facial landmark predictor")
ap.add_argument("-v", "--video", type = str, default = "", help = "path to input video file")

args = vars(ap.parse_args())


#define two constancts, one for the eye aspect ratio to indicate blink 
#and then a second constant for the number of consecutive frames the
#eye must be below the threshold

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

#initialize the frame counter and total number of blinks. COUNTER  is the total number of successive 
#frames that have an eye aspect ratio less than EYE_AR_THRESH  
#while TOTAL  is the total number of blinks that have taken place 
#while the script has been running.

COUNTER = 0
TOTAL = 0