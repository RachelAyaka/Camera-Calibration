import cv2
import numpy as np
import time
import math


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
##time
pTime = 0
cTime = 0
count = 0
speed = 0
lspeed = 0
acc = 0

cur_frame = []
prev_frame = []
lc_frame = []
lp_frame = []
curSp = []
preSp = []


while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    #frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #param: scalefactor, munNeighbors
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        #draw rectangle: top left, bottom right, color, line thickness
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 5)
        #roi = region of interest
        #find the pixel area of the image that repretents our face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]     #modify the original frame

        #center of the face
        cx = int((x+x+w)/2)
        cy = int((y+y+h)/2)
        cur_frame = (cx,cy)

        #left of the face
        lx = int(x)
        ly = int((y+y+h)/2)
        lc_frame = (lx,ly)

        #create scale for converstion from pixels into cm
        #w=19.5m -> scale=19.5m/w
        scale = 0.195/w
        #print(scale)

        if count > 2:
            #print("CUR FRAME: ", lc_frame)
            #print("PREV FRAME: ", lp_frame)
            distance = math.hypot(prev_frame[0]-cur_frame[0], prev_frame[1]-cur_frame[1])
            #make pixels into m
            distance = distance*scale 
            speed = distance*int(fps)  #m/sec
            #print(fps)
            curSp = speed

            ldistance = math.hypot(lp_frame[0]-lc_frame[0], lp_frame[1]-lc_frame[1])
            ldistance = ldistance*scale
            lspeed = int(ldistance)*int(fps)      #cm/sec

            if count > 3:
                #find the acceleration
                #print("CUR SPEED", curSp)
                #print("PRE SPEED", preSp)
                spDiff = abs(curSp-preSp)
                acc = spDiff*int(fps) #(m/s/s) = m/s^2
                #acc = int(acc)
                #print(acc)

    #setup status box
    cv2.rectangle(frame, (0,0), (275,73), (245,117,16), -1)
    #speed of center of face
    cv2.putText(frame, 'CENTER ', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, str(speed), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, 
                (255, 255, 255), 2, cv2.LINE_AA)

    #speed of left of face
    #cv2.putText(frame, 'LEFT (CM/SEC)', (105,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                (0, 0, 0), 1, cv2.LINE_AA)
    #cv2.putText(frame, str(lspeed), (100,60), cv2.FONT_HERSHEY_SIMPLEX, 2, 
    #            (255, 255, 255), 2, cv2.LINE_AA)

    #acceleration of the center of the face
    cv2.putText(frame, 'ACC (CM/SEC^2)', (105,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, str(acc), (100,60), cv2.FONT_HERSHEY_SIMPLEX, 2, 
                (255, 255, 255), 2, cv2.LINE_AA)
    #print(speed)
    #print(speed, " : ", acc)

    #head mass = 3.5kg
    #force = mass * acceleration (kg*m/s^2) = N
    force = 4.9*acc
    force = int(force)  
    print("Force: ", force)
    

    cv2.imshow('frame', frame)

    #make a copy of the points
    prev_frame = cur_frame
    lp_frame = lc_frame
    preSp = curSp

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()