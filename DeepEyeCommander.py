from Face_Detection.FaceDetection import FaceDetector, rect_to_bb, shape_to_np
from collections import Counter
import numpy as np
import cv2
import os
import tensorflow as tf


class DeepEyeCommander(object):
    def __init__(self, camera=cv2.VideoCapture(0), model=None, image_size=100):
        self.camera = camera
        self.face_detector = FaceDetector()
        self.model = model
        self.frame = None
        self.image_size = image_size
    
    def _preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.flip(gray, 1)
        return processed_frame

    def _extract_eye(self, frame):
        img = frame
        face_detected = self.face_detector.detect(img)
        success = None
        if face_detected == False:
            success = False
            return success, None
        else:
            success = True
            landmarks = self.face_detector.landmarks
            y1 = landmarks.part(18).y
            x1 = landmarks.part(18).x - 5 
            y2 = landmarks.part(29).y
            x2 = landmarks.part(29).x - 15
            crop_img = img[y1:y2, x1:x2]
            resized_img = cv2.resize(crop_img, (self.image_size, self.image_size)) 
            reshaped_img = resized_img.reshape((self.image_size,self.image_size,1))
            return success, reshaped_img
    
    def _predict_frame(self, frame):
        img = frame
        batch = np.expand_dims(img,0)
        prediction = self.model.predict_classes(batch)[0]
        return prediction

    def run_demo(self):
        classes = ['center', 'down', 'left', 'right', 'up']
        n = 3
        stack = []
       
        while self.camera.isOpened():
            success, frame = self.camera.read()
            self.frame = frame
            flipped = cv2.flip(frame, 1)
            # Stop if no video input
            if not success:
                break
            # process the frame
            self.frame = self._preprocess_frame(self.frame)
            
            eye_success, eye_img = self._extract_eye(self.frame)

            if eye_success == False:
                continue
            else:
                prediction = self._predict_frame(eye_img)
                if len(stack) == n:
                    stack.insert(0,prediction)
                    stack.pop()
                else:
                    stack.append(prediction)
            
            decision = Counter(stack).most_common(1)[0][0]
            label = classes[decision]
            color = (252, 198, 3)

            font = cv2.FONT_HERSHEY_PLAIN
            if label == 'left':
                cv2.putText(flipped, "left", (50, 375), font , 7, color, 15)
            elif label == 'right':
                cv2.putText(flipped, "right", (900, 375), font, 7, color, 15)
            elif label == 'up':
                cv2.putText(flipped, "up", (575, 100), font, 7, color, 15)
            elif label == 'down':
                cv2.putText(flipped, "down", (500, 700), font, 7, color, 15)
            else:
                # cv2.putText(flipped, " + ", (550, 375), font, 3, color, 15)
                pass

            cv2.imshow("frame", flipped)
        
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # When everything done, release the capture
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model = tf.keras.models.load_model('./Models/mark3')
    commander = DeepEyeCommander(model=model)
    commander.run_demo()

    
