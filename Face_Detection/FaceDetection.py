import dlib
import os
import cv2


class FaceDetector(object):
    CWD = os.getcwd()
    MODEL_PATH = CWD + "/models/shape_predictor_68_face_landmarks.dat"

    def __init__(self):
        self._face_detector = dlib.get_frontal_face_detector()
        self._landmark_detector = dlib.shape_predictor(self.MODEL_PATH)
        self.eye_right = None
        self.eye_left = None
        self.face_box = None
        self.landmarks = None

    # method for detecting faces and returning coordinates
    def detect(self, frame):
        _results = self._face_detector(frame)
        faceDetected = len(_results) != 0
        # if there is only one face present
        if faceDetected:
            # box is the dlib rectangle object for each face
            box = _results[0]
            # these are the individual coordinates of the face bounding box
            self.face_box = [box.top(), box.bottom(), box.left(), box.right()]
            # dlib landmarks object
            landmarks = self._landmark_detector(image=frame, box=box)
            self.landmarks = landmarks 

            self.eye_left = [
                (self.landmarks.part(i).x, self.landmarks.part(i).y) for i in range(36, 42)
            ]
            self.eye_right = [
                (self.landmarks.part(i).x, self.landmarks.part(i).y) for i in range(42, 48)
            ]
            return True
        else:
            return False


if __name__ == "__main__":

    pass
