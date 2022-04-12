import cv2
import numpy as np
from datetime import datetime
from os import path, listdir, mkdir
from shutil import rmtree
from argparse import ArgumentParser
from lib import *
import re


DEBUG_WINDOW_TITLE = 'Debug parameters'


# CLI definition
parser = ArgumentParser(description='Vinyl Cover Classifier')
parser.add_argument('--camera', action='store_true', help='Use camera instead of input images to do real-time classification')
parser.add_argument('--debug', action='store_true', help='Show debug window')
parser.add_argument('--verbose', action='store_true', help='Verbose output')
parser.add_argument('--data', type=str, default='./data/original', help='Path to data directory/file used to match potential vinyl covers (Can be a folder or a single image)')
parser.add_argument('--input', type=str, default='./data/test', help='Path to data directory/file used to match potential vinyl covers (Can be a folder or a single image)')
parser.add_argument('--output', type=str, default='./results', help='Path to output directory')
parser.add_argument('--size', type=int, default=512, help='Size to resize images to while in pipeline. (Lower = Faster, Higher = More accurate)')
args = parser.parse_args()
sampleSize = args.size


# State
data_imgs = []
input_imgs = []


def run_pipeline(img, confidence_threshold=.35):

    aspect_ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(aspect_ratio * sampleSize), sampleSize))

    # img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale image is only used for segmented image matching, not for the image processing within this pipeline
    img_grayscale = img

    # 2) Canny image
    threshold1 = cv2.getTrackbarPos('Canny threshold 1', DEBUG_WINDOW_TITLE)
    threshold1 = threshold1 if not threshold1 == -1 else 127
    threshold2 = cv2.getTrackbarPos('Canny threshold 2', DEBUG_WINDOW_TITLE)
    threshold2 = threshold2 if not threshold2 == -1 else 127
    imgCanny = cv2.Canny(img, threshold1, threshold2)

    # 3 & 4) Morphology images
    threshold3 = cv2.getTrackbarPos('Morph mask', DEBUG_WINDOW_TITLE)
    threshold3 = threshold3 if threshold3 > 0 else 12
    kernel = np.ones((threshold3, threshold3), np.uint8)
    imgMorph = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv2.morphologyEx(imgMorph, cv2.MORPH_OPEN, kernel)

    minArea = cv2.getTrackbarPos('Area threshold min', DEBUG_WINDOW_TITLE)
    minArea = minArea if not minArea == -1 else 512
    maxArea = cv2.getTrackbarPos('Area threshold max', DEBUG_WINDOW_TITLE)
    maxArea = maxArea if not maxArea == -1 else 50000
    shapes = get_shapes(imgMorph, minArea, maxPoints=0)

    # 5 & 6) Highlighted shapes image & Highlighted album image
    imgResult = img.copy()
    imgContour = img.copy()
    remaining_data_imgs = data_imgs.copy()
    matched_imgs = {}
    for points in shapes:
        x, y, w, h = cv2.boundingRect(points)
        cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 0, 255), 20)
        cv2.putText(imgContour, f'Points: {len(points)}', (x, y - 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
        for filename, data_img in remaining_data_imgs:
            name = re.sub(r'\.[^\.]+$', '', filename)
            _, confidence, _, _, projection_matrix = feature_matching(data_img, img_grayscale[y:y + h, x:x + w])
            if confidence > confidence_threshold:
                remaining_data_imgs.remove((filename, data_img))
                if args.verbose and not args.camera:
                    print(f'Found \'{name}\' ({round(confidence * 100)}%)')
                points = get_image_location(data_img, projection_matrix)
                matched_imgs[name] = (img[y:y + h, x:x + w], [p[0] for p in points], (x, y, w, h))
                break
    for name, (segment, points, (x, y, w, h)) in matched_imgs.items():
        cv2.imwrite(f'{args.output}/{name}.jpg', segment)
        imgResult[y:y + h, x:x + w] = highlight(segment, points)


    imgStack = imgs_to_stack(0.8, ([imgCanny, imgMorph], [imgContour, imgResult]))
    cv2.imshow('Vinyl Cover Classifier', imgStack)
    return matched_imgs


cooldown = datetime.now()
def change(_):
    global cooldown
    if cooldown == 0:
        cooldown = datetime.now()
    elif (datetime.now() - cooldown).seconds > 1:
        cooldown = datetime.now()
        imgIdx = cv2.getTrackbarPos('Image', DEBUG_WINDOW_TITLE)
        imgIdx = imgIdx if imgIdx > 0 else 0
        print(f'Image {imgIdx} ({input_imgs[imgIdx][0]})')
        run_pipeline(input_imgs[imgIdx][1])


def show_debug_console(useCamera):
    cv2.namedWindow(DEBUG_WINDOW_TITLE, cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(DEBUG_WINDOW_TITLE, 512, 64)
    cv2.createTrackbar('Canny threshold 1', DEBUG_WINDOW_TITLE, 127, 255, change)
    cv2.createTrackbar('Canny threshold 2', DEBUG_WINDOW_TITLE, 127, 255, change)
    cv2.createTrackbar('Morph mask', DEBUG_WINDOW_TITLE, 12, 128, change)
    cv2.createTrackbar('Area threshold min', DEBUG_WINDOW_TITLE, 512, sampleSize * sampleSize, change)
    cv2.createTrackbar('Area threshold max', DEBUG_WINDOW_TITLE, sampleSize, sampleSize * sampleSize, change)
    if useCamera == 0:
        cv2.createTrackbar('Image', DEBUG_WINDOW_TITLE, 0, len(input_imgs) - 1, change)


def rm_output_dir():
    if not path.exists(args.output):
        print(f'Output directory does not exist. ({args.output})')
        return
    rmtree(args.output)


format_re = r'\.' + '|\.'.join(['jpg', 'jpeg', 'png', 'bmp', 'pgm'])
data_names = sorted([x for x in listdir(args.data) if re.search(format_re, x)])
if len(data_names) == 0:
    print(f'No images found in data directory {args.data})')
    exit(1)
if args.verbose:
    print(f'Loading {len(data_names)} data images')
data_imgs = [(file, cv2.imread(path.join(args.data, file))) for file in data_names]
if args.verbose:
    print(f'Loaded {len(data_imgs)} data images')


# Create window in the center of the screen
if args.debug:
    show_debug_console(args.camera)


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
        run_pipeline(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:  # Update-based pipeline run
    format_re = r'\.' + '|\.'.join(['jpg', 'jpeg', 'png', 'bmp', 'pgm'])
    input_names = sorted([x for x in listdir(args.input) if re.search(format_re, x)])
    if len(input_names) == 0:
        print(f'No images found in test directory ({args.input})')
        exit(1)
    if args.verbose:
        print(f'Loading {len(input_names)} input images')
    input_imgs = [(file, cv2.imread(path.join(args.input, file))) for file in input_names]
    if args.verbose:
        print(f'Loaded {len(input_imgs)} input images')
    cv2.destroyWindow(DEBUG_WINDOW_TITLE)
    show_debug_console(0)

    rm_output_dir()
    mkdir(args.output)
    run_pipeline(input_imgs[0][1])

    print('\n(Press Ctrl+C to exit)')
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cv2.destroyAllWindows()