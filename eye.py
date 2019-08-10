import cv2 
import numpy as np
import dlib
import time

cap = cv2.VideoCapture(0) 

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while 1: 
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        font = cv2.FONT_HERSHEY_SIMPLEX

        right_eye = np.array([[]],np.int32) 

        for i in range(36,42):
            right_eye = np.append(right_eye,np.array([landmarks.part(i).x,landmarks.part(i).y]))
            if i==37 or i==40:
                right_eye = np.append(right_eye, np.array([(landmarks.part(i).x+landmarks.part(i+1).x)//2,(landmarks.part(i).y+landmarks.part(i+1).y)//2]))

        left_eye = np.array([[]],np.int32)

        for i in range(42,48):
            left_eye = np.append(left_eye,np.array([landmarks.part(i).x,landmarks.part(i).y]))
            if i==43 or i==46:
                left_eye = np.append(left_eye, np.array([(landmarks.part(i).x+landmarks.part(i+1).x)//2,(landmarks.part(i).y+landmarks.part(i+1).y)//2]))

        flag = 1

        blink_l = left_eye.reshape(-1,2)[6,1] - left_eye.reshape(-1,2)[3,1]
        blink_r = right_eye.reshape(-1,2)[6,1] - right_eye.reshape(-1,2)[3,1]
        if  blink_l <= 7 and blink_r <= 7:

            if flag == 0:
                tm = time.time()
                flag = 1
            else:
                if time.time() - tm > 0.6:
                    cv2.putText(frame,"Alert", (50,200),font,1,(0,0,220),2)

            cv2.putText(frame,"Blink", (50,50),font,1,(0,220,220),2)
            cv2.polylines(frame,[right_eye.reshape(-1,1,2)],True,(0,0,255))
            cv2.polylines(frame,[left_eye.reshape(-1,1,2)],True,(0,0,255))
        else:
            flag = 0
            tm = time.time()
            cv2.polylines(frame,[left_eye.reshape(-1,1,2)],True,(0,255,0))
            cv2.polylines(frame,[right_eye.reshape(-1,1,2)],True,(0,255,0))

    cv2.imshow('img',frame) 
    k = cv2.waitKey(30) & 0xff

    if k == 27: 
        break

cap.release() 
cv2.destroyAllWindows() 