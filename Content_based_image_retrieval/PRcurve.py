import numpy as np
from query import find_similar_by_color_histograms


def PRcurve(imgs_paths, imgs_color_hists, query_color_hist, topk, query_ground_truth_paths):
    """Calculate PR-curve.

    Args:
        imgs_paths: list of strings
        imgs_color_hists: numpy array float32
        query_color_hist: numpy array float32
        topk: int
        query_ground_truth_paths: list of strings

    Return:
        curve: numpy array float32, of shape [topk, 2]

    Note:
        tp - true positives
        fp - false positives
        tn - true negatives
        fn - false negatives

        rec - recall
        prec - precision

        gt - ground truth
    """
    def precision(tp, fp):
        return tp/(tp+fp)

    def recall(tp, fn):
        return tp/(tp+fn)

    def false_negatives(gt_size, tp):
        return gt_size - tp

    def false_positives(sample_size, tp):
        return sample_size - tp

    # usually we would use different cutoffs on distances, but there is really no point here,
    # because we only have another distinctive point on Precision-Recall curve, when we take another image.
    gt_size = len(query_ground_truth_paths)  # fixed number
    tp = np.zeros(topk)                      # vector containing values for each cutoff (number of returned images)

    ordered_paths = find_similar_by_color_histograms(imgs_paths, imgs_color_hists, query_color_hist, topk)

    # find indices of occurrences of ground truth values in full ordered output
    tp_idx = np.where(np.in1d(ordered_paths, query_ground_truth_paths))[0]

    # leverage the fact that number of true positives changes only at moments we determined above
    for idx in tp_idx:
        tp[idx:] += 1

    sample_size = np.arange(1, topk+1)       # vector containing values for each cutoff, except 0 for which precision is not well defined

    fp = false_positives(sample_size, tp)
    fn = false_negatives(gt_size, tp)
    rec = recall(tp, fn)
    prec = precision(tp, fp)

    curve = np.stack((rec, prec), axis=1)
    return curve


if __name__ == "__main__":
    # run from project folder (Content_based_image_retrieval).
    # assumed that project folder contains folder imgs with test images in it.
    # construct PR-curve for 5 image queries
    # repeat pretty much process from query.py with tweak at the end
    import os
    import cv2
    import matplotlib.pyplot as plt  # as we need to plot PR-curves, and opencv is not even remotely good for plotting
    from hist import calcColorHistsCV2, calcColorHistCV2

    imgs_folder = 'imgs'
    output_img_folder = 'report_imgs'
    histSize = 256
    ranges = [0, 256]
    query_imgs_paths = ['imgs/ukbench00004.jpg', 'imgs/ukbench00040.jpg', 'imgs/ukbench00060.jpg', 'imgs/ukbench00588.jpg', 'imgs/ukbench01562.jpg']

    # paths to images that are ground truth responses for respective queries
    imgs_ground_truth_paths = [['imgs/ukbench00004.jpg', 'imgs/ukbench00005.jpg', 'imgs/ukbench00006.jpg', 'imgs/ukbench00007.jpg'],
                               ['imgs/ukbench00040.jpg', 'imgs/ukbench00041.jpg', 'imgs/ukbench00042.jpg', 'imgs/ukbench00043.jpg'],
                               ['imgs/ukbench00060.jpg', 'imgs/ukbench00061.jpg', 'imgs/ukbench00062.jpg', 'imgs/ukbench00063.jpg'],
                               ['imgs/ukbench00588.jpg', 'imgs/ukbench00589.jpg', 'imgs/ukbench00590.jpg', 'imgs/ukbench00591.jpg'],
                               ['imgs/ukbench01560.jpg', 'imgs/ukbench01561.jpg', 'imgs/ukbench01562.jpg', 'imgs/ukbench01563.jpg']]

    # reuse functionality of find_similar_by_color_histograms, but take the whole dataset
    # with smaller values we can plot part of PR-curve
    topk = 2000

    imgs_paths = [os.path.join(imgs_folder, path) for path in sorted(os.listdir(imgs_folder))]
    imgs_color_hists = calcColorHistsCV2(imgs_paths, histSize=histSize, ranges=ranges)

    query_imgs = [cv2.imread(path, cv2.IMREAD_COLOR) for path in query_imgs_paths]
    query_color_hists = [calcColorHistCV2(img, mask=None, histSize=histSize, ranges=ranges) for img in query_imgs]

    query_PR_curves = [PRcurve(imgs_paths, imgs_color_hists, query, topk, gt) for query, gt in zip(query_color_hists, imgs_ground_truth_paths)]

    for i, curve in enumerate(query_PR_curves):
        plt.step(curve[:, 0], curve[:, 1], color='r', where='post')
        plt.title('Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        plt.savefig(f'{output_img_folder}/pr_curve_query_{i+1}')

        # flush plot
        plt.clf()
        plt.cla()
