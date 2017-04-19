import numpy as np
import math
import cv2
import settings

def add_recent_centers(num, lst, arrayToTakeCentersFrom, smooth):
    print('inside add_recent_centers lst is :', lst)
    if bool(len(arrayToTakeCentersFrom)):
        # print('we are inside add_recent_centers arrayToTakeCentersFrom not none and is: ', arrayToTakeCentersFrom)
        # add centers from this array
        i = len(lst)
        while num > len(lst):
            if len(arrayToTakeCentersFrom) > smooth:
                lst.append(arrayToTakeCentersFrom[-smooth:][i])
            else:
                lst.append(arrayToTakeCentersFrom[i])
            if i > 1:
                i -= 1
        print('lst outbound: ', lst)
        return lst
    else:
        for i in reversed(range(len(lst))):
            if len(lst) >= num:
                return lst
            else:
                lst.append(lst[i])
                print('lst is: ', lst)
        print('lst outbound: ', lst)
        return lst


def moving_average_scale(a):
    """
    This function takes in a given array (a) and performs mean normalization
    """
    mean = np.mean(a)
    std = np.std(a)

    if -2 <= std <= 2:
        return a

    good_points = []
    for item in a:
        if mean - 3 * std <= item <= mean + 3 * std:
            good_points.append(item)
        else:
            good_points.append(mean)

    mean = np.mean(good_points)
    std = np.std(good_points)
    # now grab standard scaling list
    standard = [((x - mean) / std) for x in good_points]
    results = [int(a - b) for a, b in zip(good_points, standard)]
    return results

def draw_window_box(mask, window_width, window_height, window_centroid, level):
    box = np.zeros_like(mask)
    window_start_vertical = int(mask.shape[0] - (level + 1) * (window_height))
    window_stop_vertical = int(mask.shape[0] - (level * window_height))

    window_start_horizontal = max(0, int(window_centroid - window_width))
    window_stop_horizontal = min(int(window_centroid + window_width), mask.shape[1])

    box[window_start_vertical: window_stop_vertical, window_start_horizontal: window_stop_horizontal] = 1
    return box

def draw_lines_weighted(left_points, right_points, mask):
    """
    input: left_points : binary mask for left lane line
    right_points: binary mask for right lane line
    Draw lines according to binary image mask weight
    """
    template = np.asarray(left_points + right_points).astype(np.uint8)
    tmp = np.zeros((template.shape[0], template.shape[1], 3)).astype(np.uint8)
    # make window red

    tmp[:, :, 0] = np.zeros_like(template)
    tmp[:, :, 1] = template[:, :]
    tmp[:, :, 2] = tmp[:, :, 0]

    combine_mask = np.array(cv2.merge((mask, mask, mask))).astype(np.uint8)
    weighted = cv2.addWeighted(combine_mask, 1, tmp, 0.5, 0.0)
    return weighted, mask
