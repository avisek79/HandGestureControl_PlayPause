import cv2 as cv
import mediapipe as mp
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

#Hand Detection using mediapipe Hand module
mp_hand = mp.solutions.hands        #access mediapipe's hand module
hands= mp_hand.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)    #initailize hand detection module
mp_draw = mp.solutions.drawing_utils

video_capture = cv.VideoCapture(0)

#Initialize Volume control using pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast (interface, POINTER(IAudioEndpointVolume))

#Get Volume range 
volume_range = volume.GetVolumeRange()
min_Vol, max_Vol = volume_range[0], volume_range[1]

while True:
    #read frame from webcam
    correctness, frame = video_capture.read()
    if not correctness:
        break

    #Mediapipe works on RGB so convert it to RGB
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BAYER_BG2RGB)

    #process frame with mediapipe
    result = hands.process(frame_rgb)

    cv.imshow("handGesture COntrol", result)

    if cv.waitKey(1) == ord("q"):
        break

video_capture.release()
cv.destroyAllWindows()




