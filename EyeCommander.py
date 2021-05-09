import cv2
import numpy as np
import dlib
from Face_Detection.FaceDetection import FaceDetector
from Frame_Pre_Processing.FramePreProcessing import preProcess
from ROI_Isolation.ROI_Isolation import Isolate_ROI
from Eye_Tracking.Eye_tracking import eyeCenterTracking
from CV_Classification.Eye_directions import Classifier

class EyeCommander(object):
    def __init__(self, camera=cv2.VideoCapture(0)):
        self.camera = camera
        self.face_detector = FaceDetector()
        self.rightEyeClassifier = Classifier()
        self.leftEyeClassifier = Classifier()
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.processed_frame = None
        self.frame = None
        self.face_detected = False
        self.face_box = None
        self.eye_left = None
        self.eye_right = None
        self.eye_right_cnt = None
        self.eye_right_center = None
        self.eye_left_cnt = None
        self.eye_left_center = None
        self.frame_count = 0

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
            self.eye_right, drawFigures=True
        )
        self.eye_left_cnt, self.eye_left_center, self.eye_left = eyeCenterTracking(
            self.eye_left, drawFigures=True
        )
        return None

    def make_classification(self):
        if self.frame_count <= 100:
            if 0 <= self.frame_count <= 40:
                cv2.putText(self.frame, "Look at the center area of your monitor after the count to 3", (50, 100), self.font, 2, (0, 0, 255), 3)
            elif self.frame_count <= 60:
                cv2.putText(self.frame, "1", (50, 100), self.font, 2, (0, 0, 255), 3)
            elif self.frame_count <= 80:
                cv2.putText(self.frame, "2", (50, 100), self.font, 2, (0, 0, 255), 3)
            elif self.frame_count <= 100:
                cv2.putText(self.frame, "3", (50, 100), self.font, 2, (0, 0, 255), 3)
        elif self.frame_count <= 180:
            cv2.putText(self.frame, "Keep Looking!", (50, 100), self.font, 2, (0, 0, 255), 3)
            self.leftEyeClassifier.findCenterAverage(self.frame_count, self.eye_left_center, self.eye_left_cnt)
        else:
            if self.frame_count <= 200:
               cv2.putText(self.frame, "Done!", (50, 100), self.font, 2, (0, 0, 255), 3)
            self.leftEyeClassifier.classify(self.frame_count, self.eye_left_center, self.eye_left_cnt)
        return self.leftEyeClassifier.direction

    def run_demo(self):
        while self.camera.isOpened():
            success, self.frame = self.camera.read()
            # Stop if no video input
            if not success:
                break
            
            # Class Function Calls
            processed_frame = self.process_image(self.frame)
            faceDetected = self.detect_face(processed_frame)

            if faceDetected == True:
                try:
                    self.track_eyes(processed_frame)
                    result = self.make_classification()
                    print(result)
                except:
                    pass
                cv2.imshow("frame", self.frame)
                cv2.imshow("eye_left", self.eye_left)
                cv2.imshow("eye_right", self.eye_right)
            else:
                cv2.imshow("frame", self.frame)
                print("no face detected")
            self.frame_count += 1
            # Wait for a key event
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # When everything done, release the capture
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    pass
