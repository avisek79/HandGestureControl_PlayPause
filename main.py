import cv2 as cv
import mediapipe as mp
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

#Hand Detection using mediapipe Hand module
mp_hand = mp.solutions.hands        #access mediapipe's hand module
hands= mp_hand.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)    #initailize hand detection module
mp_drawing = mp.solutions.drawing_utils

video_capture = cv.VideoCapture(0)

#Initialize Volume control using pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast (interface, POINTER(IAudioEndpointVolume))

#Get Volume range 
volume_range = volume.GetVolumeRange()
min_Vol, max_Vol = volume_range[0], volume_range[1] #Minimum and Maximum volume level

while True:
    #read frame from webcam
    correctness, frame = video_capture.read()
    if not correctness:
        break

    #Mediapipe works on RGB so convert it to RGB
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    #process frame with mediapipe
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks: #if hands are detected in frame
        for hand_landmarks in result.multi_hand_landmarks:  #iterate through each detected hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hand.HAND_CONNECTIONS)
            thumb_tip = hand_landmarks.landmark[mp_hand.HandLandmark.THUMB_TIP] #Thumb tip landmark
            index_tip = hand_landmarks.landmark[mp_hand.HandLandmark.INDEX_FINGER_TIP] #Index Tip Landmark

        #convert landmark positions to pixel coordinates
        h,w,c= frame.shape  #Get shape, height, width and Channel of the frame
        thumb_x, thumb_y = int(thumb_tip.x*w), int(thumb_tip.y*h)    #Convert Thumb coordinates to pixels
        index_x, index_y = int(index_tip.x*w), int(index_tip.y*h)    #COnvert index coordinates to pixels

        #Calculate the distance between thumb and index finger
        distance = np.hypot(index_x - thumb_x, index_y - thumb_y)

        #Map distance between thumb and index finger
        vol = np.interp(distance, [30,300],[min_Vol,max_Vol])   #Calculate the Eucledian Space
        volume.SetMasterVolumeLevel(vol, None)  #Set the system volume to calculate level

        #Draw Landmarks and connection in frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hand.HAND_CONNECTIONS)  #Draw Hand Landmarks
        cv.circle(frame, (thumb_x,thumb_y),10,(2555,0,0), cv.FILLED)    #Draw circle on the thumb tip
        cv.circle(frame, (index_x,index_y),10,(255,0,0), cv.FILLED)     #Draw circle on the index finger tip
        cv.line(frame, (thumb_x,thumb_y), (index_x, index_y), (255,0,0),3)  #Draw line between thumb and index finger



    cv.imshow("handGesture Control", frame)

    if cv.waitKey(1) == ord("q"):
        break

video_capture.release()
cv.destroyAllWindows()




