import cv2
import numpy as np
import dlib
import os
import time
import threading
from Frame_Pre_Processing.FramePreProcessing import preProcess
from ROI_Isolation.ROI_Isolation import Isolate_ROI
from Face_Detection.FaceDetection import FaceDetector
from Eye_Tracking.Eye_tracking import eyeCenterTracking
from CV_Classification.Eye_directions import Classifier 
# Module settings
FPS_LIMIT = 20
OUTPUT_MODE = 1
# Used for the FPS limiter
frame_count = 0
font = cv2.FONT_HERSHEY_SIMPLEX
# Open capturing device
# TODO: Ability to switch device based on calibration results
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# Get camera framerate and log to console
FRAME_RATE = int(cap.get(cv2.CAP_PROP_FPS))
print('Camera FPS:', FRAME_RATE)
font = cv2.FONT_HERSHEY_PLAIN
rightEyeClassifier = Classifier()
leftEyeClassifier = Classifier()
def countdown():     
        if 0 <= frame_count	 <= 40:
           cv2.putText(gray, "Prepare to look at the center area of your monitor after the count to 3", (50, 100), font, 2, (0, 0, 255), 3)
        elif frame_count <= 60:
           cv2.putText(gray, "1", (50, 100), font, 2, (0, 0, 255), 3)
        elif frame_count <= 80:
           cv2.putText(gray, "2", (50, 100), font, 2, (0, 0, 255), 3)
        elif frame_count <= 100:
           cv2.putText(gray, "3", (50, 100), font, 2, (0, 0, 255), 3)



while cap.isOpened():
    # Capture frame-by-frame
    success, frame = cap.read()

    # Stop if no video input
    if not success:
        break

    # FPS limiter
    if OUTPUT_MODE == 1:
        try:
            gray = preProcess(frame, clipLimit = 3, tileGridSize = (11,11), kernelSize = 11, blurType = 0, threshold = False)
            faces = FaceDetector()
            faces = faces.detect(gray)
            for face in faces["face_box"]:
                if 0 <= frame_count <= 100:
                   countdown()
                else:
                    left_eye = Isolate_ROI(faces["left_eye"], gray)
                    right_eye = Isolate_ROI(faces["right_eye"], gray)
                    right_eye_cnt, right_eye_center, right_eye =  eyeCenterTracking(right_eye, drawFigures = True)
                    left_eye_cnt, left_eye_center, left_eye = eyeCenterTracking(left_eye, drawFigures = True)
                    leftEyeClassifier.increaseAccumulators(left_eye_center, left_eye_cnt)
                    rightEyeClassifier.increaseAccumulators(right_eye_center, right_eye_cnt)
                cv2.imshow("face", gray)
                cv2.imshow("left_eye", left_eye)
                cv2.imshow("right_eye", right_eye)
        except:
            pass
        frame_count += 1
        leftEyeClassifier.findCenterAverage(frame_count)
        rightEyeClassifier.findCenterAverage(frame_count)
        leftEyeClassifier.classify(frame_count)
        rightEyeClassifier.classify(frame_count)
        if(leftEyeClassifier.direction != '' and rightEyeClassifier.direction != ''):
            print(leftEyeClassifier.direction, rightEyeClassifier.direction)
    # Wait for a key event
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()