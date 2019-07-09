import numpy as np


def colorHistsDistanceL2(color_hist_1, color_hist_2):
    """Calculate Euclidean distance (L2) between two color histograms.

    Args:
        color_hist_1: numpy array float32, color histogram (of the first image)
        color_hist_2: numpy array float32, color histogram (of the second image)

    Return:
        distance: float, Euclidian distance between histograms.

    Note: histograms treated here as vectors (Euclidean distance between vectors).
    For this to make sense they should be of the same length (same histSize), and belong to same vector space (same ranges).
    """
    distance = np.sqrt(np.square(color_hist_1 - color_hist_2).sum())
    return distance


def colorHistsMultipleDistancesL2(color_hists_array, color_hist):
    """Calculate Euclidean distances (L2) between array of color histograms and another color histogram.

    Args:
        color_hists_array: numpy array float32, color histograms of shape [number_of_color_histograms, histogram_length]
        color_hist: numpy array float32, color histogram

    Return:
        distances: numpy array float32, Euclidian distances between array of color histograms and given color histogram.

    Note: histograms treated here as vectors (Euclidean distance between vectors).
    For this to make sense they should be of the same length (same histSize), and belong to same vector space (same ranges).
    """
    distance = np.sqrt(np.square(color_hists_array - color_hist[np.newaxis]).sum(axis=1))
    return distance


if __name__ == "__main__":
    # sanity check
    h1 = np.array([5, 0, 3, 4], dtype=np.float32)
    h2 = np.array([4, 1, 2, 3], dtype=np.float32)
    c_distance = colorHistsDistanceL2(h1, h2)
    gt_distance = 2.0   # ground truth
    print("Do distances match?\n", c_distance == gt_distance)
    # if you see True printed everything works fine

    h_arr = np.array([[5, 0, 3, 4], [1, 4, 5, 6]], dtype=np.float32)
    h = np.array([4, 1, 2, 3], dtype=np.float32)
    c_distances = colorHistsMultipleDistancesL2(h_arr, h)
    gt_distances = np.array([2.0, 6.0])
    print("Do multiple distances match?\n", c_distances == gt_distances)
    # if you see True printed everything works fine
