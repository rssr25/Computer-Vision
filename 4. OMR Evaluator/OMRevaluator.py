import imutils
import cv2 as cv
import numpy as np
import argparse
from imutils.perspective import four_point_transform
from imutils import contours

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the input image")
args = vars(ap.parse_args())

#defining the correct answer key
ANSWER_KEY = {0:1, 1:4, 2:0, 3:3, 4:1}

#load the image, convert to grayscale, blur it out, then find edges.
image = cv.imread(args["image"])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5,5), 0)
edged = cv.Canny(blurred, 75, 200)

#find the contours in the edge map, then initialize the contour that corresponds to the document
cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None

#ensure that atleast one contour was found
if len(cnts) > 0:
    #sort contours according to their size in decreasing order
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)
    
    #loop over the sorted contours
    for c in cnts:
        #approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)

        #if approximated contour has four points then we can assume we have the paper
        if len(approx) == 4:
            docCnt = approx
            break
        else:
            print "[INFO] No contours detected."

#apply a four point perspective transform to both the original image and the grayscaled image to objtain a 
#birds-eye view of the paper (top-down)

paper = four_point_transform(image, docCnt.reshape(4,2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

#apply Otsu's thresholding method to binarize the warped piece of paper
thresh = cv.threshold(warped, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

#this ninarization will allow us to again apply contour extraction techniques to find each
#of the bubbles in the exam

#find the contours in the thresholded image, then initialize the list of contours that correcpond to questions
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionCnts = []

#loop over the contours
for c in cnts:
    #compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
    (x, y, w, h) = cv.boundingRect(c)
    ar = w / float(h)

    #in order to label the contour as a question, region should be sufficiently wide, 
    #sufficiently tall, and have an aspect ration approximately equal to 1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)

# cv.drawContours(paper, questionCnts, -1, (0, 255, 0), 2)
# cv.imshow("qConts", paper)
# cv.waitKey(0)

#GRADING STARTS HERE
#sort the question contours top-to-bottom, then initialize the total number of correct answers
questionCnts = contours.sort_contours(questionCnts, method = "top-to-bottom")[0]
correct = 0

#wach question has 5 possible answers, to loop over the question in batched of 5
for(q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    #sort the contours for the currect question from left to right, then
    #initialize the index of the bubbled answer
    cnts = contours.sort_contours(questionCnts[i:i+5])[0]
    bubbled = None

    #loop over the sorted contours
    for (j, c) in enumerate(cnts):
        #construct a mask that reveals only the current "bubble" for the question
        mask = np.zeros(thresh.shape, dtype = "uint8")
        cv.drawContours(mask, [c], -1, 255, -1)

        #apply the mask to the threshold image, then count the number of 
        #non-zero pixels in the bubble area
        mask = cv.bitwise_and(thresh, thresh, mask = mask)
        total = cv.countNonZero(mask)

        #if the current total has a larger number of total non-zero pixels
        #then we are examining the currently bubbled-in answer
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)


    #look up for the answer in the answer key
    #initialize the contour color and the index of the correct answer
    color = (0, 0, 255)
    k = ANSWER_KEY[q]

    #check to see if the bubbled answer is correct
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1
        
        #draw the outlineof the correct answer on the test
    cv.drawContours(paper, [cnts[k]], -1, color, 3)

#grab the test taker
score = (correct / 5.0) * 100
print ("[INFO] score: {:.2f}%".format(score))
cv.putText(paper, "{:.2f}%".format(score), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
cv.imshow("Original", image)
cv.imshow("Exam", paper)
cv.waitKey(0)
