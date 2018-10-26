#import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to deep learning segmentation model")
ap.add_argument("-c", "--classes", required=True,
	help="path to .txt file containing class labels")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-l", "--colors", type=str,
	help="path to .txt file containing colors for labels")
ap.add_argument("-w", "--width", type=int, default=500,
	help="desired width (in pixels) of input image")
args = vars(ap.parse_args())

#load the class label names
CLASSES = open(args["classes"]).read().strip().split("\n")

#if a colors file was supplied, load it from the disk
if args["colors"]:
	COLORS = open(args["colors"]).read().strip().split("\n")
	COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
	COLORS = np.array(COLORS, dtype="uint8")

#otherwise we need to randomly generate rgb colors for each class label
else:
	# initialize a list of colors to represent each class label in
	# the mask (starting with 'black' for the background/unlabeled
	# regions)
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
		dtype="uint8")
	COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

#initialize the legend visualization
legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype = "uint8")
# loop over the class names + colors
for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
	# draw the class name + color on the legend
	color = [int(c) for c in color]
	cv2.putText(legend, className, (5, (i * 25) + 17),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),
		tuple(color), -1)

#load our serialized model from the disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(args["model"])

# load the input image, resize it, and construct a blob from it,
# but keeping mind mind that the original input image dimensions
# ENet was trained on was 1024x512
image = cv2.imread(args["image"])
image = imutils.resize(image, width=args["width"])
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0,
	swapRB=True, crop=False)

#perform a forward pass using the segmentation model
net.setInput(blob)
start = time.time()
output = net.forward()
end = time.time()

#show the amount of time inference took
print("[INFO] inference took {:.4f} seconds".format(end - start))

# infer the total number of classes along with the spatial dimensions
# of the mask image via the shape of the output array
(numClasses, height, width) = output.shape[1:4]

# our output class ID map will be num_classes x height x width in
# size, so we take the argmax to find the class label with the
# largest probability for each and every (x, y)-coordinate in the
# image
classMap = np.argmax(output[0], axis=0)

# given the class ID map, we can map each of the class IDs to its
# corresponding color
mask = COLORS[classMap]

# resize the mask and class map such that its dimensions match the
# original size of the input image (we're not using the class map
# here for anything else but this is how you would resize it just in
# case you wanted to extract specific pixels/classes)
mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
	interpolation=cv2.INTER_NEAREST)
classMap = cv2.resize(classMap, (image.shape[1], image.shape[0]),
	interpolation=cv2.INTER_NEAREST)
 
# perform a weighted combination of the input image with the mask to
# form an output visualization
output = ((0.4 * image) + (0.6 * mask)).astype("uint8")
 
# show the input and output images
cv2.imshow("Legend", legend)
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)