import cv2
image = cv2.imread("example_01.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", image)
cv2.waitKey(0)
blur = cv2.GaussianBlur(image, (9,9), 5)
cv2.imshow("Image2", blur)
cv2.waitKey(0)