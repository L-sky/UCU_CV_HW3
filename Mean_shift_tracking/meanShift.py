# https://fr.wikipedia.org./wiki/Camshift

import numpy as np
import cv2

DELTA = 1e-7  # float literal, alias for "small enough to cause troubles with division"


def meanShift(prob_image, window, stop_criteria):
    """Update (tracking) window using mean-shift algorithm.

    Args:
        prob_image: numpy array, of probabilities that given pixel belongs to tracked object
        window: == bounding box: x,y-coordinates of left-top corner, width, height; enclosing prior tracked region
        stop_criteria: dict, where 'max_iter' - maximum number of iterations to run mean shift updates, so that we do not stuck eternally, 'epsilon' - max shift value under which we assume convergence.

    Return:
        ret: bool, whether process converged (True), or has been stopped after maximum number of iterations (False)
        window: updated bounding box
    """
    max_iter = stop_criteria['max_iter']
    epsilon = stop_criteria['epsilon']

    prob_image_height, prob_image_width = prob_image.shape

    ret = False
    for i in range(max_iter):
        # retrieve relevant values
        x, y, w, h = window

        # enforce correct range to avoid random errors
        y1 = np.maximum(y, 0)
        y2 = np.minimum(y+h, prob_image_height)
        x1 = np.maximum(x, 0)
        x2 = np.minimum(x+w, prob_image_width)

        # select region of interest
        roi = prob_image[y1:y2, x1:x2]

        # find center of masses in region of interest (in relative system of coordinates)
        M = cv2.moments(roi)

        # if there is not enough intensity in roi to reliably calculate center of mass, then most likely we lost track
        if M['m00'] < DELTA:
            # try the whole probability image to regain track
            M = cv2.moments(prob_image)

            # if probability low for the whole image, well, nothing to do, keep the last known (tracking) window
            # otherwise take the hint
            if M['m00'] < DELTA:
                break
            else:
                # make sure local coordinate system treated as global (as it should be)
                x = 0
                y = 0

        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # return to global system of coordinates, when respectively x or y <= 0, global and local axes coincide
        if x1 > 0:
            cx = cx + x1
        if y1 > 0:
            cy = cy + y1

        # convert float center of masses to bbox of fixed size defined by integers (need only x, y really)
        new_x = np.round(cx - 0.5*w).astype(np.int32)
        new_y = np.round(cy - 0.5*h).astype(np.int32)
        window = (new_x, new_y, w, h)

        # how large was shift?
        shift = np.sqrt(np.square(x-new_x) + np.square(y-new_y))

        if shift < epsilon:
            ret = True
            break

    return ret, window
