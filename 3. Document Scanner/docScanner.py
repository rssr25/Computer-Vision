#import the necessary packages
from helpFunctions.transform import four_point_transform
import argparse
import numpy as np
import cv2 as cv
import imutils
from skimage.filters import threshold_local

#constructing the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
args = vars(ap.parse_args())

########## STEP 1: EDGE DETECTION ############

#load the image and compute the ratio of the old height to the new height, clone it, and resize it
image = cv.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

#convert the image to grayscale, blur it and find edges in the image
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5,5), 0)
edged = cv.Canny(gray, 75, 200)

cv.imshow("Original", image)
cv.imshow("Gray", gray)
cv.imshow("Edged", edged)
cv.waitKey(0)

########### STEP 2: FINDING CONTOURS ##########

#find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]

#loop over the contours 
for c in cnts:
	#approximate the contour
	peri  = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.02 * peri, True)

	#if our approximated contour has four points, then we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
	else:
		print "[INFO] No Contours detected"

cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv.imshow("Outline", image)
cv.waitKey(0)

########## STEP 3: APPLY THE PERSPECTIVE TRANSFORM AND THRESHOLD ###########

#apply the four point transform to obtain a top-down view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4,2) * ratio)

#convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

#show the original and scanned images
print("STEP 3: Apply perspective transform")
cv.imshow("Original", imutils.resize(orig, height = 650))
cv.imshow("Scanned", imutils.resize(warped, height = 650))
cv.waitKey(0)