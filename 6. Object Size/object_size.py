#import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2 as cv

def midpoint(ptA, ptB):
	return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the input image")
ap.add_argument("-w", "--width", type = float, required = True, help = "width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

#load the image, convert to grayscale, blur it slightly
image = cv.imread(args["image"])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (7,7), 0)

#perform edge detection, then perform dilation + erosion to close gaps in between object edges
edged = cv.Canny(gray, 50, 100)
edged = cv.dilate(edged, None, iterations = 1)
edged = cv.erode(edged, None, iterations = 1)

#find contours in the edge map
cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

#sort the contours from left to right and initialize the 'pixel per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

#loop over the contours individually
for c in cnts:
	#if the contour is not sufficiently large, ignore it
	if cv.contourArea(c) < 100:
		continue

	#compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv.minAreaRect(c)
	box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
	box = np.array(box, dtype = "int")

	#order the points in the contour such that they appear in top-left, top-right, bottom-right, bottom-left
	#order, then draw the outline of the rotated bounding box

	box = perspective.order_points(box)
	cv.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	#loop over the original points and draw them
	for (x, y) in box:
		cv.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	#unpack the ordered bounding box, then compute the midpoint between the top-left and top-right coordinates,
	#followed by the midpoint between the bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	#compute the midpoint between the top-left and bottom-left points,
	#followed by the midpoints between top-right and bottom-right points
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	#draw the midpoints on the image
	cv.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	#draw lines between the midpoints
	cv.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
	cv.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

	#compute the euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	#if pixels per metric has not been initialized, then compute it as the ratio of pixels
	#to supplied metric (in this case, inches)

	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	#compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	#draw the object sizes on the image
	cv.putText(orig, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
	cv.putText(orig, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

	#show the output image
	cv.imshow("Image", orig)
	cv.waitKey(0)
