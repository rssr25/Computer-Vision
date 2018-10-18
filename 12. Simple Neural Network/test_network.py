#import the necessary packages
from __future__ import print_function
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

def image_to_feature_vector(image, size = (32, 32)):
	return cv2.resize(image, size).flatten()

#argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help = "path to output model file")
ap.add_argument("-t", "--test-images", required=True, help = "path to the directory of testing images")
ap.add_argument("-b", "--batch-size", type = int, default = 32, help = "size of mini-batches passed to network")
args = vars(ap.parse_args())

#initializing the class labels
CLASSES = ["cat", "dog"]

#load the network
print("[INFO] loading network architecture and weights...")
model = load_model(args["model"])
print("[INFO] testing on images in {}".format(args["test_images"]));

#loop over out testing images
for imagePath in paths.list_images(args["test_images"]):
	#load the image, resize to 32x32 and extract features from it
	print("[INFO] classifying {}".format(imagePath[imagePath.rfind("/") + 1:]))
	image = cv2.imread(imagePath)
	features = image_to_feature_vector(image) / 255.0
	features = np.array([features])

	#classify the image using our extracted features and pre-trained
	#neural network

	probs = model.predict(features)[0]
	prediction = probs.argmax(axis = 0)

	#draw the class and probability on the test image and display it on our screen
	label = "{}: {:.2f}%".format(CLASSES[prediction],
		probs[prediction] * 100)
	cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 255, 0), 3)
	cv2.imshow("Image", image)
	cv2.waitKey(0)