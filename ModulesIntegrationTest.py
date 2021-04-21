import cv2
import numpy as np
import dlib
import os
import time
from Frame_Pre_Processing.FramePreProcessing import preProcess
from ROI_Isolation.ROI_Isolation import Isolate_ROI
from Face_Detection.FaceDetection import FaceDetector
from Eye_Tracking.Eye_tracking import eyeCenterTracking
#from CV_Classification.Eye_directions import classify 

# Module settings
FPS_LIMIT = 20
OUTPUT_MODE = 1

# Used for the FPS limiter
start_time = time.time()
frame_count = 0 # pylint: disable=invalid-name

# Open capturing device
# TODO: Ability to switch device based on calibration results
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
# Get camera framerate and log to console
FRAME_RATE = int(cap.get(cv2.CAP_PROP_FPS))
print('Camera FPS:', FRAME_RATE)

while cap.isOpened():
    # Capture frame-by-frame
    success, frame = cap.read()

    # Stop if no video input
    if not success:
        break

    # FPS limiter
    current_time = time.time()
    if (current_time - start_time) > (1 / FPS_LIMIT):
        if OUTPUT_MODE == 1:
            try:	
                gray = preProcess(frame, clipLimit = 3, tileGridSize = (11,11), kernelSize = 11, blurType = 0, threshold = False)
                faces = FaceDetector()
                faces = faces.detect(gray)
                for face in faces["face_box"]:
                    left_eye = Isolate_ROI(faces["left_eye"], gray)
                    right_eye = Isolate_ROI(faces["right_eye"], gray)
                    right_eye_center, right_eye = eyeCenterTracking(right_eye, drawFigures = True)
                    left_eye_center, left_eye = eyeCenterTracking(left_eye, drawFigures = True)
                    cv2.imshow("face", gray)
                    cv2.imshow("left_eye", left_eye)
                    cv2.imshow("right_eye", right_eye)
            except:
                pass
        elif OUTPUT_MODE == 2:
            if not os.path.exists('frames'):
                os.makedirs('frames') # Create ./frames if it does not exist
            cv2.imwrite('frames/frame%d.jpg' % frame_count, frame) # Save frame as JPEG file
            print('Saved frame:', frame_count)
            frame_count += 1
        start_time = time.time() # Reset time

    # Wait for a key event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()