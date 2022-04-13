from datetime import datetime
from os import path, listdir, mkdir
from shutil import rmtree
import json
import numpy as np
import argparse
import cv2


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
    all_matches = bf.match(original_descriptors, test_descriptors)

    all_matches = sorted(all_matches, key=lambda x: x.distance)  # Orb matches by distance
    matches = all_matches[:threshold]

    # print(f'Good matches: {len(matches)}/{len(all_matches)}')

    original_points = np.float32([original_key_points[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    test_points = np.float32([test_key_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    projection_matrix, mask = cv2.findHomography(original_points, test_points, cv2.RANSAC, 5.0) # Map out outliers
    inliersMask = mask.ravel().tolist()
    matches = [match for match, isInlier in zip(matches, inliersMask) if isInlier] # Only keep inliers
    confidence = len(matches) / threshold

    return (matches, confidence, original_key_points, test_key_points, projection_matrix)


def get_image_location(image, projection_matrix):
    height, width = image.shape[:2]
    corners = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(corners, projection_matrix)
    return dst


def highlight(image, points, label = ''):
    pts = [np.int32([[p] for p in points])]
    color = (0, 0, 255)
    alpha = .25
    shapes = np.zeros_like(image, np.uint8)
    cv2.fillPoly(shapes, pts, color, lineType=cv2.LINE_AA)
    result_img = image.copy()
    mask = shapes.astype(bool)
    result_img[mask] = cv2.addWeighted(result_img, 1 - alpha, shapes, alpha, 0)[mask]
    result_img = cv2.polylines(result_img, pts, True, color, 12, lineType=cv2.LINE_AA)
    
    cv2.putText(result_img, label, (points[0][0], points[0][1] - 20), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
    return result_img


if __name__ == '__main__':
    extensions = ['jpg', 'jpeg', 'png', 'pgm']
    results_path = './results'

    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--original-dir", type=str)
    parser.add_argument("--test-img", type=str)
    args = parser.parse_args()

    # Date in seconds
    now = datetime.now().timestamp()

    original_data_dir = args.original_dir
    files = [file for file in listdir(original_data_dir) if file.split('.')[-1] in extensions]

    original_imgs = [(file, cv2.imread(path.join(original_data_dir, file))) for file in files]
    test_img = cv2.imread(args.test_img)

    # Clear results directory
    if path.exists(results_path):
        rmtree(results_path)
    mkdir(results_path)

    # Create json file
    with open(path.join(results_path, 'results.json'), 'a') as f:
        json.dump([], f)
        f.close()
    
    cropped_imgs = []
    cropped_path = path.join(results_path, "cropped_results")
    mkdir(cropped_path)
    # Loop through images
    result_img = test_img.copy()
    for file, original_img in original_imgs:
        test_extension = args.test_img.split('.')[-1]
        name = ''.join(file.split('.')[:-1])
        (matches, confidence, original_key_points, test_key_points, projection_matrix) = feature_matching(original_img, test_img)

        print(f'{name}: {round(confidence * 100)}% confidence')
        if confidence < .2:
            continue

        points = get_image_location(original_img, projection_matrix)

        point_arr = [int(x) for x in np.array(points).ravel()]
        
        (x1, y1, x2, y2, x3, y3, x4, y4) = point_arr
        cropped_im = test_img[y1:y3, x1:x3]

        result = {
            'name': name,
            'confidence': confidence,
            'points': [(point_arr[x * 2], point_arr[x * 2 + 1]) for x in range(int(len(point_arr) / 2))]
        }
        data = json.load(open(path.join(results_path, 'results.json'), 'r'))
        f.close()
        data.append(result)
        json.dump(data, open(path.join(results_path, 'results.json'), 'w'), indent=2)

        annotated_img = cv2.drawMatches(original_img, original_key_points, test_img, test_key_points, matches, None, **dict(matchColor=(255, 0, 0), flags=2))

        cv2.imwrite(f'./results/report_{name}.{test_extension}', annotated_img)
        cv2.imwrite(f'./results/cropped_results/{name}.{test_extension}', cropped_im)

    # Render results
    results = json.load(open(path.join(results_path, 'results.json')))
    for result in results:
        result_img = highlight(result_img, result['points'], result['name'])

    cv2.imwrite('./results/result.jpg', result_img)

    duration = datetime.now().timestamp() - now
    print(f'\nTime elapsed: {round(duration)} seconds')

    # Display results
    cv2.imshow('result', result_img)
    print("\n(press any key to exit window)")
    cv2.waitKey()
    cv2.destroyAllWindows()
