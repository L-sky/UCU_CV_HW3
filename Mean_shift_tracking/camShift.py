# https://fr.wikipedia.org./wiki/Camshift

import numpy as np
import cv2

from meanShift import meanShift

MARGIN = 5  # number of pixels from each side to step outside from initial window
DELTA = 1e-7
MIN_SIDE_LENGTH = 10  # how low we allow width or height to be


def camShift(prob_image, window, stop_criteria):
    """Update (tracking) window using cam-shift algorithm.

    Args:
        prob_image: numpy array, of probabilities that given pixel belongs to tracked object
        window: == bounding box: x,y-coordinates of left-top corner, width, height; enclosing prior tracked region
        stop_criteria: dict, where 'max_iter' - maximum number of iterations to run mean shift updates, so that we do not stuck eternally, 'epsilon' - max shift value under which we assume convergence.

    Return:
        ret: bool, whether process converged (True), or has been stopped after maximum number of iterations (False)
        window: updated bounding box
    """
    # run mean-shift algorithm until convergence
    ret, window = meanShift(prob_image, window, stop_criteria)

    # if mean-shift indeed converged, resize (tracking) window, otherwise just return output of mean-shift
    if ret:
        x, y, w, h = window

        # expand window
        x = x - MARGIN
        y = y - MARGIN
        w = w + 2*MARGIN
        h = h + 2*MARGIN

        roi = prob_image[y:y+h, x:x+w]
        M = cv2.moments(roi)

        # if it suddenly is smaller, then we have a lot of trouble, so just keep to the output of mean-shift
        if M['m00'] > DELTA:
            # mu stands for central(!) moments, while m for moments
            theta = np.arctan2(2*M['mu11'], M['mu20'] - M['mu02'] + np.sqrt(4*np.square(M['mu11']) + np.square(M['mu20'] - M['mu02'])))
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            I_max = M['mu20'] * np.square(cos_theta) + 2*M['mu11'] * cos_theta * sin_theta + M['mu02'] * np.square(sin_theta)
            I_min = M['mu20'] * np.square(sin_theta) + 2*M['mu11'] * cos_theta * sin_theta + M['mu02'] * np.square(cos_theta)

            # major (a) and minor (b) axis of ellipse
            a = 4 * np.sqrt(I_max / M['m00'])
            b = 4 * np.sqrt(I_min / M['m00'])

            # if major axes appeared to be shorter
            if a < b:
                a, b = b, a
                # theta <- pi/2 - theta; sin(pi/2 - theta) = cos(theta); cos(pi/2 - theta) = sin(theta)
                cos_theta, sin_theta = sin_theta, cos_theta

            # calculate new width and height
            w = np.maximum(a*cos_theta, b*sin_theta)
            h = np.maximum(a*sin_theta, b*cos_theta)

            # article comes with a recommendation to enlarge window by 20% (I assume each side), so be it
            w = 1.2 * w
            h = 1.2 * h

            # enforce data type
            w = np.round(w).astype(np.int64)
            h = np.round(h).astype(np.int64)

            # enforce at least minimal size
            w = np.maximum(w, MIN_SIDE_LENGTH)
            h = np.maximum(h, MIN_SIDE_LENGTH)

            # pack back
            window = (x, y, w, h)

    return ret, window
