#this program runs the style transfer in a video. Bitch you can even change the styles by pressing 
#the letter 'n' for "next" <3

#import the necessary packages
from imutils.video import VideoStream
from imutils import paths
import itertools
import argparse
import imutils
import time
import cv2

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True, help = "path to directory containing neural style transfer models")
args = vars(ap.parse_args())

#grab the parth to all neural style transfer models in out 'models'
#directory, provided all models end with .t7 file extension
modelPaths = paths.list_files(args["models"], validExts = (".t7"))
modelPaths = sorted(list(modelPaths))

#generate unique IDs for each of the model paths, then combine the
#two lists togetehr
models = list(zip(range(0, len(modelPaths)), (modelPaths)))

#use the cycle function of itertools that can loop over all model
#paths, and then when the end is reached, restart again
modelIter = itertools.cycle(models)
(modelID, modelPath) = next(modelIter)

#load the neural style transfer model from disk
print("[INFO] loading style trander model...")
net = cv2.dnn.readNetFromTorch(modelPath)

#initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src = 0).start()
time.sleep(2.0)
print("[INFO] {}. {}".format(modelID + 1, modelPath))

#loop over the frames from the video file stream
while True:
	#grab the frame from the threaded video stream
	frame = vs.read()

	#resize the frame with a width of 600 pixels (maintain aspect ratio), and then
	#grab the image dimensions
	frame = imutils.resize(frame, width = 600)
	orig = frame.copy()
	(h, w) = frame.shape[:2]

	#construct blob, set input, and then forward pass that shit
	blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
	net.setInput(blob)
	output = net.forward()

	# reshape the output tensor, add back in the mean subtraction, and
	# then swap the channel ordering
	output = output.reshape((3, output.shape[2], output.shape[3]))
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	output /= 255.0
	output = output.transpose(1, 2, 0)
 
	# show the original frame along with the output neural style
	# transfer
	cv2.imshow("Input", frame)
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `n` key is pressed (for "next"), load the next neural
	# style transfer model
	if key == ord("n"):
		# grab the next neural style transfer model model and load it
		(modelID, modelPath) = next(modelIter)
		print("[INFO] {}. {}".format(modelID + 1, modelPath))
		net = cv2.dnn.readNetFromTorch(modelPath)
 
	# otheriwse, if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

#clean that mess
cv2.destroyAllWindows()
vs.stop()