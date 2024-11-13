import os
from collections import ChainMap, OrderedDict

import cv2
import numpy as np
import pandas as pd

from configs import configs
from utils import closest_odd, check_int


def get_color_distribution(im_arr: np.ndarray, resize_to: int=150,
                           blur_kernel_size: int=20, mono: bool=False) -> OrderedDict:
    """
    Extract color statistics for a single image

    :param im_arr: Input image array in BGR format
    :type im_arr: numpy.ndarray

    :param resize_to: Target size to resize the image, maintaining the aspect ratio
    :type resize_to: int, optional

    :param blur_kernel_size: Kernel size for median blurring; must be an odd number
    :type blur_kernel_size: int, optional

    :param mono: If image is monochrome or not
    :type mono: bool, optional

    :return: Distribution of colors in the image, mapped and generalized into predefined ranges
    :rtype: OrderedDict
    """
    h, w = im_arr.shape[:2]
    if h > w:
        im_arr = cv2.rotate(im_arr, cv2.ROTATE_90_CLOCKWISE)

    target_width = min(resize_to, w)
    im_arr = _image_resize(im_arr, width=target_width)

    kernel_size = closest_odd(blur_kernel_size)
    im_arr = cv2.medianBlur(im_arr, kernel_size)

    im_arr_hsv = cv2.cvtColor(im_arr, cv2.COLOR_BGR2HSV)
    im_arr_hsv = im_arr_hsv.reshape(-1, 3).astype('int16')

    if mono:
        # supposing gray color would be 1/3 of all colors if it was rgb
        # determines the size of the color planet
        count = im_arr_hsv.shape[0]/3 # supposing gray color would be 1/3 of all colors if it was rgb
        hue = sat = 0
        value = np.mean(im_arr_hsv[:,2]) # for monochrome images only value is positive (0,0,v)
        color_groups_generalized = dict(mono=[count, hue, sat, value])
    else:
        generalize_by_hue = np.vectorize(lambda x: configs.HUE_RANGES_LOOKUP[x])

        im_arr_hsv[:, 0] = generalize_by_hue(im_arr_hsv[:, 0])
        im_arr_hsv[:, 1], sat_ranges = pd.cut(im_arr_hsv[:, 1], bins=10,
                                              right=False, labels=False, retbins=True)
        im_arr_hsv[:, 2], value_ranges = pd.cut(im_arr_hsv[:, 2], bins=10,
                                                right=False, labels=False, retbins=True)

        im_arr_hsv = im_arr_hsv[im_arr_hsv[:, 0].argsort()]

        color_groups = np.split(im_arr_hsv, np.unique(im_arr_hsv[:, 0], return_index=True)[1][1:])

        color_groups_generalized = ChainMap(*(_generalize_by_hsv(group, sat_ranges, value_ranges)
                                              for group in color_groups))

    missing_colors = set(configs.HUE_COLORS.keys()) - set(color_groups_generalized.keys())
    missing_colors = {k: [np.nan] * 4 for k in missing_colors}

    color_distribution = OrderedDict(sorted({**color_groups_generalized, **missing_colors}.items()))

    return color_distribution


def check_image(img_path: str, min_size_kb: int=1, max_size_mb: int=100) -> bool:
    """
    :param img_path: The path to the image file to be checked.
    :param min_size_kb: Minimum file size in kilobytes to consider it an image
    :param max_size_mb: Maximum file size in megabytes to consider it an image
    :return: True if the file is a valid image, False otherwise.
    """

    try:
        file_size_kb = os.path.getsize(img_path) / 1024  # Convert to KB
        return min_size_kb <= file_size_kb <= (max_size_mb * 1024)
    except (FileNotFoundError, OSError):
        return False

def truncate_string(x: str, limit: int) -> str:
    """
    Apply a length limit on a string

    :param x: string to be truncated
    :param limit: max length of the string
    """
    if not isinstance(x, str):
        if check_int(x):
            return str(x)
        return x

    if len(x) <= limit:
        return x
    return f'{x[:limit]}..'

def _image_resize(image: np.ndarray, width: int=None,
                 height: int=None, inter=cv2.INTER_AREA) -> np.ndarray:
    """
    :param image: Input image to be resized.
    :param width: The desired width of the resized image
    :param height: The desired height of the resized image
    :param inter: Interpolation method used for resizing. Default is cv2.INTER_AREA.
    :return: Resized image.
    """
    (h, w, dim) = image.shape

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def _generalize_by_hsv(hsv: np.ndarray, sat_ranges: list, value_ranges: list) -> dict:
    """
    Replace hue, saturation and value in the group with mean values

    :param hsv: An array of HSV color values.
    :param sat_ranges: A list of saturation ranges.
    :param value_ranges: A list of value ranges.
    :return: A dictionary with color as the key and a list containing count, hue generalizer,
             saturation generalizer, and value generalizer as the value.
    """
    sv_combinations, sv_combination_counts = np.unique(np.column_stack((hsv[:, 1], hsv[:, 2])),
                                                 return_counts=True, axis=0)
    most_frequent_combination = sv_combinations[np.argmax(sv_combination_counts)]
    most_frequent_sat_range, most_frequent_value_range = most_frequent_combination

    sat_generalizer = sat_ranges[most_frequent_sat_range: most_frequent_sat_range + 2].mean()
    value_generalizer = value_ranges[most_frequent_value_range: most_frequent_value_range + 2].mean()

    count = hsv.shape[0]
    color = configs.HUE_COLORS_REV[hsv[0][0]]
    hue_generalizer = configs.HUE_COLORS_MEAN[color]
    return {color: [count, hue_generalizer, sat_generalizer, value_generalizer]}


def _get_progress_string(pbar, idx: int, total: int, prefix: str) -> str:
    """
    Assemble progress info in tqdm style

    :param pbar: tqdm.tqdm object
    :param idx: iteration index
    :param total: total number of iterations
    :param prefix; custom prefix
    """

    percentage = idx / total * 100

    elapsed = pbar.format_dict['elapsed']
    elapsed_str = pbar.format_interval(elapsed)

    rate = pbar.format_dict["rate"]
    if not rate:
        rate = 1

    remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
    remaining_str = pbar.format_interval(remaining)

    text = f"""{prefix}: {percentage:.1f}% {idx}/{total} 
                [{elapsed_str}<{remaining_str}  {rate:.0f} it/s]"""

    return text





