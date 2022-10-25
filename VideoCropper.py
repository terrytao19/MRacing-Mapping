import cv2
import numpy as np
import math

cap = cv2.VideoCapture('C:\\Users\\terry\\MICHIGAN\\MRacing\\YOLOv7-cone\\yolov7-cone\\KIT22d.mp4')
vid_writer = cv2.VideoWriter('C:\\Users\\terry\\MICHIGAN\\MRacing\\videos\\KIT22d.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                          (1280, 720))
i=0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(i)
        cv2.rectangle(frame, [0, 0], [int(1280), int(720/2)], [0, 0, 0], -1)
        vid_writer.write(frame)
        i +=1
    if i == 4464:
        break
cap.release()
vid_writer.release()