from datetime import datetime
import numpy as np
import argparse
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--test-img", type=str)
    args = parser.parse_args()

    now = datetime.now().timestamp()

    test_img = cv2.imread(args.test_img)

    Z = test_img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)

    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 4, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    result_img = res.reshape((test_img.shape))

    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)

    duration = datetime.now().timestamp() - now
    print(f'\nTime elapsed: {round(duration)} seconds')

    cv2.imwrite('./results/result.jpg', result_img)
