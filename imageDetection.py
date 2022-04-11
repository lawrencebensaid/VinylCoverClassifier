import cv2
import numpy as np
from datetime import datetime
from os import path, listdir, mkdir
from shutil import rmtree
import json
import argparse

# Settings
windowTitle = 'Parameters'

sampleSize = 512

# Camera
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# State
img_target = None
testDir = './data/test'
testImgs = []


def feature_matching(original_img, test_img):
    threshold = 50  # Amount of matches used for perspective transform

    # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    # original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Method 1: SIFT
    # sift = cv2.SIFT_create(nfeatures=100000, edgeThreshold=0)
    # test_key_points, test_descriptors = sift.detectAndCompute(test_img, None)
    # original_key_points, original_descriptors = sift.detectAndCompute(original_img, None)

    # Method 2: ORB
    orb = cv2.ORB_create(nfeatures=100000, WTA_K=2, edgeThreshold=0, patchSize=25)
    original_key_points, original_descriptors = orb.detectAndCompute(original_img, None)
    test_key_points, test_descriptors = orb.detectAndCompute(test_img, None)

    # print(f'original\n  points: {len(original_key_points)}\n  descriptors: {len(original_descriptors)}\ntest\n  points: {len(test_key_points)}\n  descriptors: {len(test_descriptors)}')

    # Match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if original_descriptors is None or test_descriptors is None:
        return (None, 0, [], [], None)
    all_matches = bf.match(original_descriptors, test_descriptors)

    all_matches = sorted(all_matches, key=lambda x: x.distance)  # Orb matches by distance
    if len(all_matches) < 5:
        return (None, 0, [], [], None)
    matches = all_matches[:threshold]

    # print(f'Good matches: {len(matches)}/{len(all_matches)}')

    original_points = np.float32([original_key_points[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    test_points = np.float32([test_key_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    projection_matrix, mask = cv2.findHomography(original_points, test_points, cv2.RANSAC, 5.0) # Map out outliers
    inliersMask = mask.ravel().tolist()
    matches = [match for match, isInlier in zip(matches, inliersMask) if isInlier] # Only keep inliers
    confidence = len(matches) / threshold

    return (matches, confidence, original_key_points, test_key_points, projection_matrix)


def imgs_to_stack(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0]. shape [:2]:
                    imgArray[x][y] = cv2.resize (imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def get_shapes(img, minArea=0, maxArea=50000, maxPoints=0):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    shapes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea and area < maxArea:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if maxPoints == 0 or len(approx) <= maxPoints:
                shapes.append(approx)
    return shapes


def get_image_location(image, projection_matrix):
    height, width = image.shape[:2]
    corners = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(corners, projection_matrix)
    return dst


def highlight(image, points, label = ''):
    pts = [np.int32([[p] for p in points])]
    color = (0, 255, 0)
    alpha = .25
    shapes = np.zeros_like(image, np.uint8)
    cv2.fillPoly(shapes, pts, color, lineType=cv2.LINE_AA)
    result_img = image.copy()
    mask = shapes.astype(bool)
    result_img[mask] = cv2.addWeighted(result_img, 1 - alpha, shapes, alpha, 0)[mask]
    result_img = cv2.polylines(result_img, pts, True, color, 12, lineType=cv2.LINE_AA)
    
    # cv2.putText(result_img, label, (points[0][0], points[0][1] - 20), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
    return result_img


def run():

    _, img = cap.read()
    # imageIndex = cv2.getTrackbarPos('Image', windowTitle)
    # print(f'Image {imageIndex}')
    # img = testImgs[imageIndex]

    aspect_ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(aspect_ratio * sampleSize), sampleSize))

    # 2) Canny image
    threshold1 = cv2.getTrackbarPos('Canny threshold 1', windowTitle)
    threshold2 = cv2.getTrackbarPos('Canny threshold 2', windowTitle)
    imgCanny = cv2.Canny(img, threshold1, threshold2)

    # 3 & 4) Morphology images
    threshold3 = cv2.getTrackbarPos('Morph mask', windowTitle)
    kernel = np.ones((threshold3, threshold3), np.uint8)
    imgMorph = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv2.morphologyEx(imgMorph, cv2.MORPH_OPEN, kernel)

    minArea = cv2.getTrackbarPos('Area threshold min', windowTitle)
    maxArea = cv2.getTrackbarPos('Area threshold max', windowTitle)
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


def update(a):
    run()


if __name__ == '__main__':
    # extensions = ['jpg', 'jpeg', 'png', 'pgm']

    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--img", type=str)
    # parser.add_argument("--test-img", type=str)
    args = parser.parse_args()

    img_target = cv2.imread(args.img)
    # testImgs = [cv2.imread(path.join(testDir, file)) for file in listdir(args.test_img)]

    # Create window in the center of the screen
    cv2.namedWindow(windowTitle, cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(windowTitle, 512, 64)
    cv2.createTrackbar('Canny threshold 1', windowTitle, 127, 255, update)
    cv2.createTrackbar('Canny threshold 2', windowTitle, 127, 255, update)
    cv2.createTrackbar('Morph mask', windowTitle, 12, 128, update)
    # cv2.createTrackbar('Image', windowTitle, 0, len(testImgs) - 1, update)
    cv2.createTrackbar('Area threshold min', windowTitle, 512, sampleSize * sampleSize, update)
    cv2.createTrackbar('Area threshold max', windowTitle, sampleSize, sampleSize * sampleSize, update)

    # run() # Enable for update-based run

    while True:

        run() # Enable for real-rime run loop

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break