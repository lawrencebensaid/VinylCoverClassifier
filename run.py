import cv2
import numpy as np
from datetime import datetime
from os import path, listdir, mkdir
from shutil import rmtree
import json
import argparse
import threading
from lib import *
import subprocess


# Settings
windowTitle = 'Parameters'

sampleSize = 512

# State
img_target = None
testDir = './data/test'
testImgs = []


# CLI definition
parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--img", type=str)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--camera", action="store_true")
parser.add_argument("--input", type=str, default='./data/test')
parser.add_argument("--results", type=str, default='./results')
args = parser.parse_args()


def runPipeline(img):

    aspect_ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(aspect_ratio * sampleSize), sampleSize))

    # 2) Canny image
    threshold1 = cv2.getTrackbarPos('Canny threshold 1', windowTitle)
    threshold1 = threshold1 if not threshold1 == -1 else 127
    threshold2 = cv2.getTrackbarPos('Canny threshold 2', windowTitle)
    threshold2 = threshold2 if not threshold2 == -1 else 127
    imgCanny = cv2.Canny(img, threshold1, threshold2)

    # 3 & 4) Morphology images
    threshold3 = cv2.getTrackbarPos('Morph mask', windowTitle)
    threshold3 = threshold3 if threshold3 > 0 else 12
    kernel = np.ones((threshold3, threshold3), np.uint8)
    imgMorph = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv2.morphologyEx(imgMorph, cv2.MORPH_OPEN, kernel)

    minArea = cv2.getTrackbarPos('Area threshold min', windowTitle)
    minArea = minArea if not minArea == -1 else 512
    maxArea = cv2.getTrackbarPos('Area threshold max', windowTitle)
    maxArea = maxArea if not maxArea == -1 else 50000
    shapes = get_shapes(imgMorph, minArea, maxPoints=0)

    # 5 & 6) Highlighted shapes image & Highlighted album image
    imgsCropped = []
    imgResult = img.copy()
    imgContour = img.copy()
    for points in shapes:
        x, y, w, h = cv2.boundingRect(points)
        cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 0, 255), 20)
        cv2.putText(imgContour, f'Points: {len(points)}', (x, y - 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
        imgCropped = img[y:y + h, x:x + w]
        imgsCropped.append(imgCropped)
        (matches, confidence, original_key_points, test_key_points, projection_matrix) = feature_matching(img_target, imgCropped)
        if confidence > 0.4:
            print(confidence)
            points = get_image_location(img_target, projection_matrix)
            imgResult[y:y + h, x:x + w] = highlight(imgCropped, [p[0] for p in points])


    imgStack = imgs_to_stack(0.8, ([imgCanny, imgMorph], [imgContour, imgResult]))
    cv2.imshow('Vinyl Cover Classifier', imgStack)


cooldown = datetime.now()
def change(_):
    global cooldown
    if cooldown == 0:
        cooldown = datetime.now()
    elif (datetime.now() - cooldown).seconds > 1:
        cooldown = datetime.now()
        imgIdx = cv2.getTrackbarPos('Image', windowTitle)
        imgIdx = imgIdx if imgIdx > 0 else 0
        print(f'Image {imgIdx}')
        runPipeline(testImgs[imgIdx])


def showDebugConsole(useCamera):
    cv2.namedWindow(windowTitle, cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(windowTitle, 512, 64)
    cv2.createTrackbar('Canny threshold 1', windowTitle, 127, 255, change)
    cv2.createTrackbar('Canny threshold 2', windowTitle, 127, 255, change)
    cv2.createTrackbar('Morph mask', windowTitle, 12, 128, change)
    cv2.createTrackbar('Area threshold min', windowTitle, 512, sampleSize * sampleSize, change)
    cv2.createTrackbar('Area threshold max', windowTitle, sampleSize, sampleSize * sampleSize, change)
    if useCamera == 0:
        cv2.createTrackbar('Image', windowTitle, 0, len(testImgs) - 1, change)


# extensions = ['jpg', 'jpeg', 'png', 'pgm']

img_target = cv2.imread(args.img)

# Create window in the center of the screen
if args.debug:
    showDebugConsole(args.camera)


if args.camera:  # Continuous pipeline run loop (Real-rime video stream)
    frameWidth = 640
    frameHeight = 480
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    print('Real-time video stream started')
    print('\n(Press Ctrl+C to exit)')
    while True:
        _, img = cap.read()
        runPipeline(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:  # Update-based pipeline run
    testImgs = [cv2.imread(path.join(testDir, file)) for file in listdir(args.input)]
    if len(testImgs) == 0:
        print('No images found in test directory')
        exit(1)
    print(f'Loaded {len(testImgs)} images')
    cv2.destroyWindow(windowTitle)
    showDebugConsole(0)

    runPipeline(testImgs[0])

    print('\n(Press Ctrl+C to exit)')
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cv2.destroyAllWindows()