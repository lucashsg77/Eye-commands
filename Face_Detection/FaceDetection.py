import dlib
import os


class FaceDetector:
    CWD = os.path.abspath(os.path.dirname(__file__))
    MODEL_PATH = CWD.replace(
        os.path.dirname(__file__), "models/shape_predictor_68_face_landmarks.dat"
    )
    # dlib detectors
    FACE = dlib.get_frontal_face_detector()
    LANDMARKS = dlib.shape_predictor(MODEL_PATH)

    def __init__(self):
        pass

    # method for detecting faces and returning coordinates
    def detect(self, frame):
        face_results = self.FACE(frame)
        # if there is only one face present
        if len(face_results) == 1:
            # box is the dlib rectangle object for each face
            box = face_results[0]
            # these are the individual coordinates of the face bounding box
            x1, y1, x2, y2 = box.left(), box.top(), box.right(), box.bottom()
            # dlib landmarks object
            landmark_results = self.LANDMARKS(image=frame, box=box)
            # building a dictionary with arrays of tuple coordinate pairs (x,y) as values
            # and face landmarks as keys
            results = {}
            results["face_box"] = [(x1, y1), (x2, y2)]
            results["left_eye"] = [
                (landmark_results.part(i).x, landmark_results.part(i).y)
                for i in range(36, 42)
            ]
            results["left_eyebrow"] = [
                (landmark_results.part(i).x, landmark_results.part(i).y)
                for i in range(17, 22)
            ]
            results["right_eye"] = [
                (landmark_results.part(i).x, landmark_results.part(i).y)
                for i in range(42, 48)
            ]
            results["right_eyebrow"] = [
                (landmark_results.part(i).x, landmark_results.part(i).y)
                for i in range(22, 27)
            ]
            return results
        else:
            return None


if __name__ == "__main__":
    pass