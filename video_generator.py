import cv2
import numpy as np
import glob
import os

directory = './datasets/oxygen-bvm'
imgs=[]


height = 480
width = 640



for trial,filename in enumerate(sorted(glob.iglob(f'{directory}/**/videodata/', recursive=True))):
    print(filename)
    imgs = []
    for frame,imagepath in enumerate(sorted(glob.iglob(f'{filename}/**/*original.jpg', recursive=True),key=os.path.getmtime)):
        imgs.append(cv2.rotate(cv2.imread(imagepath),cv2.ROTATE_180))

    out = cv2.VideoWriter(f'{filename}newvideo.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, (width,height))

    for i in range(len(imgs)):
        out.write(imgs[i])
    out.release()

    # break
