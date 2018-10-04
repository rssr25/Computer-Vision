import imutils
import cv2 as cv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input image")
args = vars(ap.parse_args())

image = cv.imread(args["image"])
#cv.imshow("Image", image);
#cv.waitKey(0)

#convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)
cv.waitKey(0)

#thresholding
thresh = cv.threshold(gray, 254, 255, cv.THRESH_BINARY_INV)[1]
cv.imshow("thresh", thresh)
cv.waitKey(0)

#detect the edges
edged = cv.Canny(gray, 30, 150)
cv.imshow("Edged", edged)
cv.waitKey(0)

# find contours (i.e., outlines) of the foreground objects in the
# thresholded image
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
output = image.copy()
 
# loop over the contours
for c in cnts:
	# draw each contour on the output image with a 3px thick purple
	# outline, then display the output contours one at a time
	cv.drawContours(output, [c], -1, (240, 0, 159), 3)
	cv.imshow("Contours", output)
	cv.waitKey(0)

#after that we do erosion and dilation using cv.erode and cv.dilate
