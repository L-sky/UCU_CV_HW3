import cv2
import numpy as np

from filterpy.kalman import KalmanFilter


def mousePos(event, x, y, flags, params):
    # update mouse position on every mouse move
    # conveniently, if no move occurs params still hold the last value, which remains relevant
    if event == cv2.EVENT_MOUSEMOVE:
        params[0] = x
        params[1] = y


def construct_measurement(x, x_prev, y, y_prev):
    # use differences of two consequent direct measurements as a velocity
    vx, vy = x - x_prev, y - y_prev
    return np.array([[x],
                     [vx],
                     [y],
                     [vy]], dtype=np.float32)


if __name__ == "__main__":
    kalman_filter = KalmanFilter(dim_x=4, dim_z=4)  # dim_x <-> [x, vx, y, vy] and dim_z <-> [x, vx, y, vy]
    dt_ms = 40                                          # aiming for 25 fps here
    dt_s = dt_ms / 1000
    fps = 1. / dt_s                                     # realistically, it would be (consistently) lower depending on speed of data processing

    # define base level of noise (rather arbitrary)
    noise_base = np.array([[20.],
                           [1.],
                           [20.],
                           [1.]])

    # define size of the window to interact with
    img_size = (720, 1024, 3)
    img_height, img_width, img_channels = img_size

    # fading of plotted image over time, lower alpha means faster fading
    alpha = 0.9                                         # 0 - instant fading, 1 - no fading
    alpha_mask = np.empty(img_size)                     # made mask an array instead of single number, so that cv2.multiply keeps all channels, and not just first (blue)
    alpha_mask[:] = alpha

    # setup initial state of display image
    img_prev = np.zeros(img_size, dtype=np.uint8)
    param = np.zeros(2, dtype=np.int32)                 # placeholder
    cv2.imshow('img', img_prev)
    cv2.setMouseCallback('img', mousePos, param)
    cv2.waitKey(delay=dt_ms)                            # make sure that the first point is a cursor position, not initial [0., 0.]

    # initiate filter
    x_prev, y_prev = param
    cv2.waitKey(delay=dt_ms)
    x, y = param                                        # need two points to define velocity
    estimate_x_prev, estimate_y_prev = None, None
    estimate_x, estimate_y = x_prev, y_prev

    # construct Initial state
    kalman_filter.x = construct_measurement(x, x_prev, y, y_prev)

    # define internal matrices
    kalman_filter.F = np.array([[1., dt_s, 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 1., dt_s],
                                [0., 0., 0., 1.]])      # State transition matrix: [x + vx*dt; vx; y + vy*dt; vy]

    kalman_filter.H = np.eye(4, dtype=np.float32)       # Measurement matrix: identity transform here
    kalman_filter.R = 1.                                # State uncertainty
    KalmanFilter.Q = np.diag([0., 3., 0., 3.])          # Process uncertainty
    kalman_filter.P = 0.                                # Covariance matrix (initial)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (img_width, img_height))
    red = (0, 0, 255)
    blue = (255, 0, 0)
    while True:
        # iterate until Esc
        key = cv2.waitKey(delay=dt_ms)
        if key == 27:
            break

        # get measurement
        x_prev, y_prev = x, y
        x, y = param
        measurement = construct_measurement(x, x_prev, y, y_prev)

        # artificially introduce noise
        noisy_measurement = measurement + np.random.randn(4, 1) * noise_base
        noisy_measurement_x, noisy_measurement_y = np.round(noisy_measurement[0][0]).astype(np.int32), np.round(noisy_measurement[2][0]).astype(np.int32)

        # apply kalman filter
        kalman_filter.predict()
        kalman_filter.update(noisy_measurement)
        estimate = np.round(kalman_filter.x).astype(np.int32)
        estimate_x_prev, estimate_y_prev = estimate_x, estimate_y
        estimate_x, estimate_y = estimate[0][0], estimate[2][0]

        # plot current noisy measurement as red dot, and change from previous estimate to current as blue line
        img = np.zeros(img_size, dtype=np.uint8)
        cv2.circle(img, (noisy_measurement_x, noisy_measurement_y), 3, red, -1)
        cv2.line(img, (estimate_x_prev, estimate_y_prev), (estimate_x, estimate_y), blue, 3)

        # push frame to video
        out.write(img_prev)

        # update of image with fading happens here
        cv2.multiply(alpha_mask, img_prev, dst=img_prev, dtype=cv2.CV_8UC3)
        cv2.add(img_prev, img, dst=img_prev, dtype=cv2.CV_8UC3)
        cv2.imshow('img', img_prev)

        # ground truth mouse cursor position versus estimate
        # print('measurement:', measurement.reshape(-1), "estimate:", estimate.reshape(-1))

    out.release()
    cv2.destroyAllWindows()

