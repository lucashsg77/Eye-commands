from Face_Detection.FaceDetection import FaceDetector
import cv2
import os
import glob
from DLCommander import DLCommander

commander = DLCommander(image_size=100)
for name in ['up','down','left','right','center']:

    files = glob.glob(f'./data/imgs/{name}/*.jpg') 
    writepath = f'./data/data/{name}'
    count = 0

    for file in files:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        success, eye = commander._extract_eye(processed_img)
        if success == False:
            continue
        else:
            cv2.imwrite(os.path.join(writepath,name+str(count)+'.jpg'), img=eye)
            count+=1

    print(f'done {name}.')
