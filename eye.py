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
    #print(faces)
    for face in faces:
        # x1,y1 = face.left(), face.top()
        # x2, y2 = face.right(), face.bottom()
        # # print(x1,x2,y1,y2)
        # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        landmarks = predictor(gray, face)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # try:
        #     if(landmarks):
        #         cv2.putText(frame,"face detected" ,(60,60), font,1,(0,244,0),2)
        # except:
        #     print()

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


        # right_eye = right_eye.reshape(-1,2)
        # left_eye = left_eye.reshape(-1,2)

        # outer_mouth = np.array([[]],np.int32)
        # for i in range(48,60):
        #     outer_mouth = np.append(outer_mouth,np.array([landmarks.part(i).x,landmarks.part(i).y]))


        # inner_mouth = np.array([[]],np.int32)
        # for i in range(60,68):
        #     inner_mouth = np.append(inner_mouth,np.array([landmarks.part(i).x,landmarks.part(i).y]))
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

        # cv2.polylines(frame,[outer_mouth.reshape(-1,1,2)],True,(0,255,0))
        # cv2.polylines(frame,[inner_mouth.reshape(-1,1,2)],True,(0,255,0))

        

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # if abs(landmarks.part(62).y - landmarks.part(66).y) > 3:
    #     cv2.putText(frame,"Mouth Opened" ,(60,60), font,1,(0,244,0),2)
    # else:
    #     cv2.putText(frame,"Mouth Closed" ,(60,60), font,1,(0,244,0),2)
    
    cv2.imshow('img',frame) 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
cap.release() 
cv2.destroyAllWindows() 