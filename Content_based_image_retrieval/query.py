import numpy as np
from L2 import colorHistsMultipleDistancesL2


def find_similar_by_color_histograms(imgs_paths, imgs_color_hists, query_color_hist, topk):
    """Retrieve from dataset images closest to given based on L2 distance between color histograms.

    Args:
        imgs_paths: list of strings
        imgs_color_hists: numpy array float32
        query_color_hist: numpy array float32
        topk: int

    Return:
        top_imgs_paths: list of strings with selected image paths
    """
    if isinstance(imgs_paths, list):
        imgs_paths = np.array(imgs_paths)

    distances = colorHistsMultipleDistancesL2(imgs_color_hists, query_color_hist)

    # get order of sorting distances in ascending order
    order = np.argsort(distances)

    # take indices corresponding to first topk closest images
    order = order[:topk]

    # retrieve closest images
    top_imgs_paths = imgs_paths[order]
    return top_imgs_paths


if __name__ == "__main__":
    # run from project folder (Content_based_image_retrieval).
    # assumed that project folder contains folder imgs with test images in it.
    # retrieve closest to the query images from dataset based on L2 distance between color histograms.
    import os
    import cv2
    from hist import calcColorHistsCV2, calcColorHistCV2


    def resize_image(image, width=None, height=None):
        """Wrapper around cv2.resize that allows to specify target size of either side and resize image while keeping aspect ratio."""
        h, w = image.shape[:2]

        if width:
            h = h * width / w
            w = width
        else:
            w = w * height / h
            h = height

        resized_image = cv2.resize(image, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        return resized_image


    imgs_folder = 'imgs'
    histSize = 256
    ranges = [0, 256]
    query_imgs_paths = ['imgs/ukbench00004.jpg', 'imgs/ukbench00040.jpg', 'imgs/ukbench00060.jpg', 'imgs/ukbench00588.jpg', 'imgs/ukbench01562.jpg']
    topk = 10
    close_images_display_height = 128

    imgs_paths = [os.path.join(imgs_folder, path) for path in sorted(os.listdir(imgs_folder))]
    imgs_color_hists = calcColorHistsCV2(imgs_paths, histSize=histSize, ranges=ranges)

    query_imgs = [cv2.imread(path, cv2.IMREAD_COLOR) for path in query_imgs_paths]
    query_color_hists = [calcColorHistCV2(img, mask=None, histSize=histSize, ranges=ranges) for img in query_imgs]

    topk_imgs_paths = [find_similar_by_color_histograms(imgs_paths, imgs_color_hists, query, topk=topk) for query in query_color_hists]

    for query_img, close_imgs_paths in zip(query_imgs, topk_imgs_paths):
        # load close images in memory and change size, so that they all fit in screen
        close_imgs = [resize_image(cv2.imread(path, cv2.IMREAD_COLOR), height=close_images_display_height) for path in close_imgs_paths]

        # concatenate images horizontally for display purposes
        close_imgs = np.concatenate(close_imgs, axis=1)

        cv2.imshow('query image', query_img)
        cv2.imshow(f'close images ({topk})', close_imgs)
        cv2.waitKey()
