
from cv2 import *
import cv2 as cv
import time


cam_port = 0
cam = VideoCapture(cam_port)
time.sleep(3)
# reading the input using the camera
result, image = cam.read()


cv.imshow("joe", image)

    # saving image in local storage
cv.imwrite("input.png", image)

