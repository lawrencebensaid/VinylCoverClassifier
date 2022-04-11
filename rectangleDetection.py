import cv2
import numpy as np

# Settings
windowTitle = 'Parameters'

# Camera
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def stackImages(scale, imgArray):
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





def getShapes(img, areaThreshold=500):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    shapes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > areaThreshold:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            shapes.append(approx)
    return shapes


def getBinaryImage(img):
    result_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result_img = cv2.threshold(result_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    mask = np.ones((128, 128), np.uint8)
    result_img = cv2.morphologyEx(result_img, cv2.MORPH_CLOSE, mask)
    mask = np.ones((32, 32), np.uint8)
    result_img = cv2.morphologyEx(result_img, cv2.MORPH_OPEN, mask)
    result_img = cv2.morphologyEx(result_img, cv2.MORPH_ERODE, mask)
    return result_img

def applyKMeans(img):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z) # convert to np.float32

    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 4, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center) # Convert back to uint8
    res = center[label.flatten()]
    result_img = res.reshape((img.shape))

    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    return result_img


def run():

    success, img = cap.read()
    # img = cv2.imread('results/CroppedImage.jpg')

    # Get aspect ratio
    aspect_ratio = img.shape[1] / img.shape[0]

    # Rescale image to width of 512. Keep the same aspect ratio.
    img = cv2.resize(img, (int(aspect_ratio * 512), 512))

    imgContour = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos('Canny threshold1', windowTitle)
    threshold2 = cv2.getTrackbarPos('Canny threshold2', windowTitle)
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    threshold3 = cv2.getTrackbarPos('Morph mask', windowTitle)
    kernel = np.ones((threshold3, threshold3), np.uint8)
    imgClosed = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, kernel)
    imgOpened = cv2.morphologyEx(imgClosed, cv2.MORPH_OPEN, kernel)
    threshold4 = cv2.getTrackbarPos('Morph mask 2', windowTitle)
    kernel = np.ones((threshold4, threshold4), np.uint8)
    imgClosed2 = cv2.morphologyEx(imgOpened, cv2.MORPH_CLOSE, kernel)


    shapes = getShapes(imgClosed2, areaThreshold=1000)

    for points in shapes:
        x, y, w, h = cv2.boundingRect(points)
        cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 0, 255), 20)
        cv2.putText(imgContour, f'Points: {len(points)}', (x, y - 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)

    imgCropped = img[y:y + h, x:x + w]

    imgStack = stackImages(0.8, ([img, imgGray, imgCanny], [imgClosed, imgOpened, imgClosed2], [imgContour, imgContour, imgCropped]))

    # cv2.imwrite('results/CroppedImage.jpg', imgCropped)

    cv2.imshow('Result', imgStack)
    cv2.moveWindow('Result', 256, 128)



def update(a):
    print('RUN')
    run()

# Create window in the center of the screen
cv2.namedWindow(windowTitle, cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow(windowTitle, 512, 64)
cv2.createTrackbar('Canny threshold1', windowTitle, 20, 255, update)
cv2.createTrackbar('Canny threshold2', windowTitle, 25, 255, update)
cv2.createTrackbar('Morph mask', windowTitle, 8, 128, update)
cv2.createTrackbar('Morph mask 2', windowTitle, 16, 128, update)

# run()

while True:

    run()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break