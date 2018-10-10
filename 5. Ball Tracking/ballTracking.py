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

#defining the lower and upper boundaries of the "green" ball in the HSV color space,
#then initialize the list of tracked points

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen = args["buffer"])

#if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

#otherwise, grab a reference to the video file
else:
	vs = cv.VideoCapture(args["video"])

#allow the camera or video file to warm up
time.sleep(2.0)

#keep looping
while True:
	#grab the current frame
	frame = vs.read()

	#handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	#if we are viewing a video and we did not grab a frame, then we have reached the end of the bideo
	if frame is None:
		break

	#resize the frame, blur it and convert it to HSV color space
	frame = imutils.resize(frame, width = 600)
	blurred = cv.GaussianBlur(frame, (11,11), 0)
	hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

	#construct a mas for the color "green", then perform a series of dilations and erosions
	#to remove any small blobs left in the mask

	mask = cv.inRange(hsv, greenLower, greenUpper)
	mask = cv.erode(mask, None, iterations = 2)
	mask = cv.dilate(mask, None, iterations = 2)

	#find contours in the mask and initialize the current
	#(x, y) center of the ball
	cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	center = None

	#proceed if at least one contour was found
	if len(cnts) > 0:
		#find the largest contour in the mask, then use it to compute the minimum enclosing
		#circle and centroid

		c = max(cnts, key = cv.contourArea)
		((x, y), radius) = cv.minEnclosingCircle(c)
		M = cv.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		#only proceed if the radius meets a minimum size
		if(radius > 10):
			#draw the circle and centroid on the frame,
			#then update the list of tracked points

			cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			cv.circle(frame, center, 5, (0, 0, 255), -1)

	#update the points queue
	pts.appendleft(center)

	#loop over the set of tracked points
	for i in range(1, len(pts)):
		#if either of the tracked points are none, ignore them
		if pts[i - 1] is None or pts[i] is None:
			continue

		#otherwise, compute the thickness of the line and draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	#show the frame to our screen
	cv.imshow("Frame", frame)
	key = cv.waitKey(1) & 0xFF

	#if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

#if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

#otherwise, release the camera
else:
	vs.release()

#destroy all windows
cv.destroyAllWindows()





 