import imutils
import cv2 as cv

image = cv.imread("jp.png")
h, w, d = image.shape

b, g, r = image[100, 100]
print("R={}, G={}, B={}".format(r, g, b))

#next we have array slicing which is done using the numpy array rows and columns
#resizing the image is also a very straightforward thing cv.resize(image, (200 200))

#using the aspect ratio and then resizing the image
#instead of manually calculating the aspect ratio we can just use imutils.resize which does it for us.
#