import cv2
import numpy as np


# own implementation of histogram
# interface based on respective opencv function
def calcHist(img, channel, mask, histSize, ranges):
    """Calculate histogram bin values.

    Args:
        img: numpy array of numbers, assumed to be representation of an image, where last dimension is channels
        channel: int, which channel of the image to consider
        mask: numpy array of bool values, which spatial part of the image to consider. If None - consider whole image
        histSize: int, number of bins
        ranges: list of two numbers, smallest (inclusive) and largest (exclusive) values to consider

    Return:
        bin_counts: numpy array float32, counts of occurrences per each bin (respective interval)
        intervals: numpy array float32, of shape [number_of_bins, 2], each row is an interval that bin covers: left side inclusive, right side exclusive
    """
    left_boundary, right_boundary = ranges
    bin_width = (right_boundary - left_boundary) / histSize

    data = img[..., channel]
    if mask:
        data = data[mask]

    # inclusive left, exclusive right
    data = data[np.logical_and(left_boundary <= data, data < right_boundary)].reshape(-1)

    ticks = np.linspace(start=left_boundary, stop=right_boundary, num=histSize + 1)  # number of bins = number of ticks - 1
    intervals = np.stack((ticks[:-1], ticks[1:]), axis=-1)

    # shift and stretch domain, so that floor gives sequence number (= bin number)
    data = ((data - left_boundary) / bin_width).astype(np.int32)

    bin_counts = np.zeros(histSize, dtype=np.float32)  # type can be int32 (makes more sense, but opencv outputs float32)
    nonzero_bin_numbers, nonzero_bin_counts = np.unique(data, return_counts=True)
    bin_counts[nonzero_bin_numbers] = nonzero_bin_counts

    return bin_counts, intervals


def calcColorHist(img, mask, histSize, ranges):
    """Same as calcHist, but go over all color channels, assuming BGR image, and concatenate histograms together."""
    hist_blue, _ = calcHist(img, channel=0, mask=mask, histSize=histSize, ranges=ranges)
    hist_green, _ = calcHist(img, channel=1, mask=mask, histSize=histSize, ranges=ranges)
    hist_red, _ = calcHist(img, channel=2, mask=mask, histSize=histSize, ranges=ranges)

    hist_color = np.concatenate((hist_blue, hist_green, hist_red))
    return hist_color


def calcColorHistCV2(img, mask, histSize, ranges):
    """Same as cv2.calcHist, but go over all color channels, assuming BGR image, and concatenate histograms together."""
    if isinstance(histSize, int):
        histSize = [histSize]

    hist_blue = cv2.calcHist([img], channels=[0], mask=mask, histSize=histSize, ranges=ranges).reshape(-1)
    hist_green = cv2.calcHist([img], channels=[1], mask=mask, histSize=histSize, ranges=ranges).reshape(-1)
    hist_red = cv2.calcHist([img], channels=[2], mask=mask, histSize=histSize, ranges=ranges).reshape(-1)

    hist_color = np.concatenate((hist_blue, hist_green, hist_red))
    return hist_color


# Dropping mask parameter in this wrapper, as not going to use any.
# Logically it should be separate mask for each image, and generating random ones is pointless.
def calcColorHists(imgs_paths, histSize, ranges):
    """Calculate color histograms for all images provided with their paths.

    Args:
        imgs_paths: list of strings, with paths to images
        histSize: int, number of bins
        ranges: list of two numbers, smallest (inclusive) and largest (exclusive) values to consider

    Return:
        color_hists: numpy array float32, of shape [number_of_images, 3*histSize]
    """
    # paths to images is a more flexible way than either path to the folder (how do we subset images?) and way less memory bounded than image arrays
    color_hists = np.stack([calcColorHist(cv2.imread(path, cv2.IMREAD_COLOR), mask=None, histSize=histSize, ranges=ranges) for path in imgs_paths])
    return color_hists


# same thing with mask parameter for opencv based implementation.
def calcColorHistsCV2(imgs_paths, histSize, ranges):
    """Calculate color histograms for all images provided with their paths.

    Args:
        imgs_paths: list of strings, with paths to images
        histSize: int, number of bins
        ranges: list of two numbers, smallest (inclusive) and largest (exclusive) values to consider

    Return:
        color_hists: numpy array float32, of shape [number_of_images, 3*histSize]
    """
    color_hists = np.stack([calcColorHistCV2(cv2.imread(path, cv2.IMREAD_COLOR), mask=None, histSize=histSize, ranges=ranges) for path in imgs_paths])
    return color_hists


if __name__ == "__main__":
    # run from project folder (Content_based_image_retrieval).
    # assumed that project folder contains folder imgs with test images in it.
    # calculate color histograms for all images using own and opencv based implementations, compare.
    import os
    imgs_folder = 'imgs'
    histSize = 256     # bigger numbers only introduce empty bins, smaller - drop resolution, while I would prefer to keep all information
    ranges = [0, 256]  # there is nothing outside this range, and again no reason to randomly shrink the range and drop information
    imgs_paths = [os.path.join(imgs_folder, path) for path in sorted(os.listdir(imgs_folder))]
    color_hists = calcColorHists(imgs_paths, histSize=histSize, ranges=ranges)
    color_hists_cv2 = calcColorHistsCV2(imgs_paths, histSize=histSize, ranges=ranges)
    print("Do color histograms calculated with own implementation and opencv-based implementation match for all images?\n", np.all(color_hists == color_hists_cv2))
    # if you see True printed, then everything works fine
