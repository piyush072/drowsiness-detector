import cv2 
import numpy as np
import dlib

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
        
        try:
            if(landmarks):
                cv2.putText(frame,"Nikal lavde, Pehli fursat me nikal" ,(60,60), font,1,(0,244,0),2)
        except:
            print()

        # left_eye = np.array([[]],np.int32) 
        # for i in range(36,42):
        #     left_eye = np.append(left_eye,np.array([landmarks.part(i).x,landmarks.part(i).y]))

        # right_eye = np.array([[]],np.int32)
        # for i in range(42,48):
        #     right_eye = np.append(right_eye,np.array([landmarks.part(i).x,landmarks.part(i).y]))

        # left_eye = left_eye.reshape(-1,2)
        # right_eye = right_eye.reshape(-1,2)

        # outer_mouth = np.array([[]],np.int32)
        # for i in range(48,60):
        #     outer_mouth = np.append(outer_mouth,np.array([landmarks.part(i).x,landmarks.part(i).y]))


        # inner_mouth = np.array([[]],np.int32)
        # for i in range(60,68):
        #     inner_mouth = np.append(inner_mouth,np.array([landmarks.part(i).x,landmarks.part(i).y]))


        # cv2.polylines(frame,[outer_mouth.reshape(-1,1,2)],True,(0,255,0))
        # cv2.polylines(frame,[inner_mouth.reshape(-1,1,2)],True,(0,255,0))
        # cv2.polylines(frame,[right_eye.reshape(-1,1,2)],True,(0,255,0))
        # cv2.polylines(frame,[left_eye.reshape(-1,1,2)],True,(0,255,0))

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # if abs(landmarks.part(62).y - landmarks.part(66).y) > 3:
    #     cv2.putText(frame,"Muh band kar nai toh makkhi ghus jayega" ,(60,60), font,1,(0,244,0),2)
    # else:
    #     cv2.putText(frame,"Mouth Closed" ,(60,60), font,1,(0,244,0),2)
    
    cv2.imshow('img',frame) 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
cap.release() 
cv2.destroyAllWindows() 