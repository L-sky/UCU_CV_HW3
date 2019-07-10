import cv2
import numpy as np
import argparse


def get_margins(side_length):
    if side_length % 2 == 0:
        left = -1 + side_length // 2
        right = side_length // 2
    else:  # == 1
        left = right = side_length // 2

    return left, right  # top, bottom respectively for height


# partial matching on edges for criteria
def SAD(img_patch, template):
    return np.nanmean(np.abs(img_patch - template))


def SSD(img_patch, template):
    return np.nanmean(np.square(img_patch - template))


def NCC(img_patch, template):
    nans = np.isnan(img_patch)
    if np.any(nans):
        template = template.copy()
        template[nans] = np.nan

    normalization = np.sqrt(np.nanmean(np.square(img_patch)) * np.nanmean(np.square(template)))
    return np.nanmean(img_patch * template) / normalization


def template_match_map(img, template, criteria):
    img_height, img_width = img.shape
    template_height, template_width = template.shape

    top_margin, bottom_margin = get_margins(template_height)
    left_margin, right_margin = get_margins(template_width)

    # make sure we skip areas outside the image
    padded_img = np.pad(img, ((top_margin, bottom_margin), (left_margin, right_margin)), mode='constant', constant_values=np.nan)

    criteria_values_map = np.zeros((img_height, img_width), dtype=np.float32)
    for y in range(img_height):
        for x in range(img_width):
            img_patch = padded_img[y:y + bottom_margin + top_margin + 1, x:x + left_margin + right_margin + 1]
            criteria_values_map[y, x] = criteria(img_patch, template)

    return criteria_values_map


def template_match_position(criteria_values_map, criteria_type, template):
    # default
    extremum = np.argmin
    if criteria_type in ['sad', 'ssd']:
        extremum = np.argmin
    elif criteria_type in ['ncc']:
        extremum = np.argmax

    cy, cx = np.unravel_index(extremum(criteria_values_map), criteria_values_map.shape)
    value = criteria_values_map[cy, cx]

    height, width = template.shape
    top_shift, _ = get_margins(height)
    left_shift, _ = get_margins(width)

    bbox = (cx-left_shift, cy-top_shift, width, height)

    return (cy, cx), value, bbox


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="string path to the image for which you would like to apply template matching")
    parser.add_argument("template", help="string path to the image to be used as template")
    parser.add_argument("criteria_type", help="which comparison method to use", choices=['sad', 'ssd', 'ncc'])
    parser.add_argument("--bbox", help="segment of template image to be actually used as template defined by coordinates of left top corner and height, width. If not supplied, the whole image to be used.", nargs=4, type=int)

    args = parser.parse_args()

    img_path = args.img
    template_img_path = args.template
    criteria_type = args.criteria_type

    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY).astype(np.float32)

    template_color = cv2.imread(template_img_path, cv2.IMREAD_COLOR)
    template = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if criteria_type == 'sad':
        criteria = SAD
    elif criteria_type == 'ssd':
        criteria = SSD
    elif criteria_type == 'ncc':
        criteria = NCC
    else:
        print("Defaulted to SSD, check if loss_type has been entered correctly!")
        criteria = SSD

    if args.bbox:
        x, y, w, h = args.bbox
        template_color = template_color[y:y + h, x:x + w]
        template = template[y:y + h, x:x + w]

    criteria_values_map = template_match_map(img, template, criteria)
    center, value, bbox = template_match_position(criteria_values_map, criteria_type, template)
    print("Center:", center, "Map value:", value, "BBox:", bbox)

    x, y, w, h = bbox

    cv2.rectangle(img_color, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.imshow('img', img_color)
    cv2.imshow('template', template_color)
    cv2.waitKey()
