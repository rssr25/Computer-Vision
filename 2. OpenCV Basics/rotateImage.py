import cv2 as cv

image = cv.imread("jp.png")
h, w = image.shape[:2]
center = (w//2, h//2)
M = cv.getRotationMatrix2D(center, 45, 1.0)
rotated = cv.warpAffine(image, M, (w,h))
cv.imshow("Image rotated", rotated)
cv.waitKey(0)

#we can also use imutils.rotate function for similar functionality
#also imutils.rotate_bound function will not clip the image after rotation

