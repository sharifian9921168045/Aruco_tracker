import cv2
import matplotlib.pyplot as plt
import numpy as np
from aruco_detector import  Marker,edge_detector
import time
# Load live camera

#num = input('please enter number :')
#cap = cv2.VideoCapture('https://192.168.169.45:8080/video')
#cap = cv2.VideoCapture('https://22.155.148.41:8080/video')
cap = cv2.VideoCapture('https://22.122.99.146:8080/video')
#cap = cv2.VideoCapture('192.168.42.2')


#cap = cv2.VideoCapture('test_video/VID_20230808_102548 (online-video-cutter.com).mp4')
#cap = cv2.VideoCapture('test_video/main_mian.mp4')

while True:
    _, first_frame = cap.read()
    first_frame = cv2.resize(first_frame, (1000, 700), interpolation=cv2.INTER_AREA)
    #cv2.imwrite('text_main.jpg',first_frame)
    break

canny = edge_detector(first_frame)

# gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# canny = cv2.Canny(blurred, 70, 215)




perspective = cv2.imread("perspective.jpg")
#f = open(f'test_log/log{num}','w')
while True:
    now = time.perf_counter()
    _,img = cap.read()
    img = cv2.resize(img,(1000,700),interpolation=cv2.INTER_AREA)
    cv2.imshow('org',img)
    marker = Marker(img,canny)
    dst = marker.crop_picture()
    img_marker = marker.detect_marker(dst)
    center_marker = marker.marker_location()
    # if center_marker != []:
    #     marker.tracker(perspective , center_marker)
    #print(center_marker,print(type(center_marker)))
    li = []

    if center_marker != [] :
        if center_marker.ndim == 1 :
            li.append(marker.calculate(center_marker[0], center_marker[1], 120, 84, [0, 0], [1000, 0], [0, 700]))
        else:
            for center in center_marker:
                 li.append(marker.calculate(center[0],center[1],120,84,[0,0],[1000,0],[0,700]))
    print(li,now)
    #f.write(f'{li} --- {now}\n')

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# cv2.imshow('persective',perspective)
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()







    # if center:cdcd
    #     print(marker.calculate(center[0],center[1],150,100,listpoint['top_left'] ,listpoint['top_right'] ,\
    #                 listpoint['buttom_left']))
    #output = pose_estimation(img,cv2.aruco.DICT_7X7_50, intrinsic_camera, distortion)

    #cv2.imshow("Image", img)
    # key = cv2.waitKey(1)
    # if key == ord("q"):
    #     break



