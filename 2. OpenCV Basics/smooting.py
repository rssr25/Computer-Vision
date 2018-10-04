import cv2 as cv
import imutils

image = cv.imread("jp.png")
blurred = cv.GaussianBlur(image, (11,11), 0)
cv.imshow("Blurred image", blurred)
cv.waitKey(0)

#next up is drawing on an image. Rectangle, circle and lines.