import imutils
import cv2 as cv

image = cv.imread("jp.png")
h, w, d = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

cv.imshow("Image", image)
cv.waitKey(0)