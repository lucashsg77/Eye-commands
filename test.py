import cv2
import numpy as np
import dlib

def print_eye(eye_points, facial_landmarks):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    #cv2.polylines(frame, [eye_region], True, (0, 0, 255), 2)
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]
    #ret,thresh = cv2.threshold(gray_eye,0, 255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #thresh2 = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
    return gray_eye


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#fps =  cap.get(cv2.CAP_PROP_FPS)
#print(w,h,fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    _, frame = cap.read()
    height, width, _ = frame.shape
    norm = np.zeros((height, width))
    aux = cv2.normalize(frame,  norm, 0, 255, cv2.NORM_MINMAX)
    frame = aux
    aux = cv2.GaussianBlur(frame, (5,5), 0)
    frame = aux
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    aux = clahe.apply(gray)
    gray = aux
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        cv2.imshow("Frame", frame)
        cv2.imshow("Right Eye",print_eye([36, 37, 38, 39, 40, 41], landmarks))
        cv2.imshow("Left Eye",print_eye([42, 43, 44, 45, 46, 47], landmarks))
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()