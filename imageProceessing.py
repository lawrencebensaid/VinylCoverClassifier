from datetime import datetime
import numpy as np
import argparse
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--test-img", type=str)
    args = parser.parse_args()

    now = datetime.now().timestamp()

    test_img = cv2.imread(args.test_img, cv2.IMREAD_GRAYSCALE)

    # result_img = cv2.threshold(result_img, 125,)
    result_img = cv2.threshold(test_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    mask = np.ones((128, 128), np.uint8)
    result_img = cv2.morphologyEx(result_img, cv2.MORPH_CLOSE, mask)
    mask = np.ones((32, 32), np.uint8)
    result_img = cv2.morphologyEx(result_img, cv2.MORPH_OPEN, mask)
    result_img = cv2.morphologyEx(result_img, cv2.MORPH_ERODE, mask)

    duration = datetime.now().timestamp() - now
    print(f'\nTime elapsed: {round(duration)} seconds')

    cv2.imwrite('./results/result_ip.jpg', result_img)
