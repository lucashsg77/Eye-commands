import cv2
import numpy as np
import dlib
from Face_Detection.FaceDetection import FaceDetector
from Frame_Pre_Processing.FramePreProcessing import preProcess
from ROI_Isolation.ROI_Isolation import *
from Eye_Tracking.Eye_tracking import *
from CV_Classification.Eye_directions import Classifier


class EyeCommander(object):
    def __init__(self, camera=cv2.VideoCapture(0)):
        self.camera = camera
        self.classifier = Classifier()
        self.face_detector = FaceDetector()

        self.processed_frame = None
        self.face_detected = False
        self.face_box = None
        self.eye_left = None
        self.eye_right = None
        self.eye_right_cnt = None
        self.eye_right_center = None
        self.eye_left_cnt = None
        self.eye_left_center = None

    def process_image(self, frame):
        processed_frame = preProcess(
            frame,
            clipLimit=3,
            tileGridSize=(11, 11),
            kernelSize=11,
            blurType=0,
            threshold=False,
        )
        self.processed_frame = processed_frame
        return processed_frame

    def detect_face(self, frame):
        face_detected = self.face_detector.detect(frame)

        if face_detected == True:
            self.eye_left = self.face_detector.eye_left
            self.eye_right = self.face_detector.eye_right
            self.face_box = self.face_detector.face_box
        return face_detected

    def track_eyes(self, frame):
        self.eye_right = Isolate_ROI(self.eye_right, frame)
        self.eye_left = Isolate_ROI(self.eye_left, frame)
        self.eye_right_cnt, self.eye_right_center, self.eye_right = eyeCenterTracking(
            self.eye_right, drawFigures=False
        )
        self.eye_left_cnt, self.eye_left_center, self.eye_left = eyeCenterTracking(
            self.eye_left, drawFigures=False
        )
        return None

    def make_classification(self, frame):
        pass

    def run_demo(self):

        while self.camera.isOpened():
            success, frame = self.camera.read()
            # Stop if no video input
            if not success:
                break
            
            # Class Function Calls
            processed_frame = self.process_image(frame)
            faceDetected = self.detect_face(processed_frame)

            if faceDetected == True:
                print(self.eye_left)
                cv2.imshow("frame", frame)
            else:
                cv2.imshow("frame", frame)
                print("no face detected")

            # Wait for a key event
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # When everything done, release the capture
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    pass
