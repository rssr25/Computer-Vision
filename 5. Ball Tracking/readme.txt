This thang can be done by following the steps below:

1. Load the video file / start the camera stream
2. grab a frame
3. resize the frame
4. blur that shit out 
5. convert to HSV color space
6. mask the color you want to track
7. erode that bitch mask
8. dilate that bitch too
9. find the contours
10. draw the circles for the required contour
11. fill a deque will the previous tracked points
12. draw the deque points for the contrails.