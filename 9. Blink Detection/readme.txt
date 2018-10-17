This program uses dlib for facial landmarking. The landmarking is used to detect the 
Number of times the user has blinked his/her eyes. 

To build our blink detector, we’ll be computing a metric called the eye aspect ratio (EAR), introduced by Soukupová and Čech in their 2016 paper, Real-Time Eye Blink Detection Using Facial Landmarks.

Unlike traditional image processing methods for computing blinks which typically involve some combination of:

Eye localization.
Thresholding to find the whites of the eyes.
Determining if the “white” region of the eyes disappears for a period of time (indicating a blink).
The eye aspect ratio is instead a much more elegant solution that involves a very simple calculation based on the ratio of distances between facial landmarks of the eyes.

This method for eye blink detection is fast, efficient, and easy to implement.