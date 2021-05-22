import cv2
import os

def dir_to_frames(inpath, writepath):
    writepath = writepath
    files = [i for i in os.listdir(inpath) if i != '.DS_Store' and i != '.ipynb_checkpoints']
    count = 1
    for f in files:
        vidpath = os.path.join(inpath, f)
        vidcap = cv2.VideoCapture(vidpath)
        success,image = vidcap.read()
        framecount = 0
        while success:
            try:
                success,image = vidcap.read()
                cv2.imwrite(writepath + str(count)+"frame%d.jpg" % framecount, image)     # save frame as JPEG file      
                print('Read a new frame: ', success)
                framecount += 1
            except:
                pass
        count+=1