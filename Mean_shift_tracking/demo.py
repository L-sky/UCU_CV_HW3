# Based on: https://docs.opencv.org/master/d7/d00/tutorial_meanshift.html

import numpy as np
import cv2
import os
import argparse

from meanShift import meanShift
from camShift import camShift


# example queries:
# ./imgs/Girl/img ./imgs/Girl/img/0001.jpg 57 21 31 45 meanshift_own 10 1
# ./imgs/BlurCar2/img ./imgs/BlurCar2/img/0001.jpg 227 207 122 99 camshift_cv2 100 1
# ./imgs/Skating2/img ./imgs/Skating2/img/0001.jpg 347 58 103 251 camshift_own 10 1
# ./imgs/BlurFace/img ./imgs/BlurFace/img/0001.jpg 246 226 94 114 meanshift_cv2 10 1
# ./imgs/Board/img ./imgs/Board/img/00001.jpg 57 156 198 173 camshift_own 10 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="string path to the folder with images to apply tracking")
    parser.add_argument("anchor", help="string path to the image to be used as template, usually first image in sequence")
    parser.add_argument("bbox", help="region of interest to track, x,y-coordinates of left-top corner, width, height", nargs=4, type=int)
    parser.add_argument("tracker", help="which tracker to use", choices=['meanshift_cv2', 'camshift_cv2', 'meanshift_own', 'camshift_own'])
    parser.add_argument("max_iter", help="maximum number of iterations inside tracker per one frame", type=int)
    parser.add_argument("epsilon", help="distance boundary for convergence in tracker", type=float)

    args = parser.parse_args()

    dataset_path = args.dataset
    anchor_img_path = args.anchor
    track_window = tuple(args.bbox)  # because opencv is very specific it should be tuple, not a list
    tracker_type = args.tracker
    max_iter = args.max_iter
    epsilon = args.epsilon

    # set tracker and stopping criteria
    tracker = None
    term_crit = None
    if tracker_type == 'meanshift_cv2':
        tracker = cv2.meanShift
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, epsilon)
    elif tracker_type == 'camshift_cv2':
        tracker = cv2.CamShift
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, epsilon)
    elif tracker_type == 'meanshift_own':
        tracker = meanShift
        term_crit = {'max_iter': max_iter, 'epsilon': epsilon}
    elif tracker_type == 'camshift_own':
        tracker = camShift
        term_crit = {'max_iter': max_iter, 'epsilon': epsilon}

    # retrieve paths to all frames
    dataset_img_paths = [os.path.join(dataset_path, img_path) for img_path in sorted(os.listdir(dataset_path))]

    # retrieve frame containing known region of interest
    anchor_img = cv2.imread(anchor_img_path, cv2.IMREAD_COLOR)

    # retrieve region of interest
    x, y, w, h = track_window
    roi = anchor_img[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # discard dark, poorly saturated regions (treated as noise)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    # calculate histogram over hue channel (step = 1 degree)
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

    # recalibrate histogram
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # display region of interest
    cv2.imshow('roi', roi)

    for i, img_path in enumerate(dataset_img_paths):
        # read next frame and convert it to probability image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply selected tracker
        ret, track_window = tracker(dst, track_window, term_crit)

        # display tracking result
        x, y, w, h = track_window
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('track', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
