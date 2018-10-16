PLEASE INSTALL dlib BEFORE PROCEEDING.

Detecting facial landmarks is therefore a two step process:

Step #1: Localize the face in the image.
Step #2: Detect the key facial structures on the face ROI.

the actual algorithm used to detect the face in the image doesn’t matter. Instead, what’s important is that through some method we obtain the face bounding box (i.e., the (x, y)-coordinates of the face in the image).

There are a variety of facial landmark detectors, but all methods essentially try to localize and label the following facial regions:

Mouth
Right eyebrow
Left eyebrow
Right eye
Left eye
Nose
Jaw

