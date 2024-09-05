from itertools import chain, combinations
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from skimage.segmentation import mark_boundaries

def get_subsets(iterable):
    """returns a list of all subsets of an iterable"""
    s = list(iterable)
    ps = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
    ps = [list(l) for l in ps]
    return ps

def generate_subsets(n):
    subsets = []
    for i in range(2 ** n):
        subset = [j for j in range(n) if (i >> j) & 1]
        subsets.append(subset)
    return subsets

def get_supersets(subset, N):
    """return all supersets, given a subset and size of the entire set"""
    super_sets = []
    all_subsets = get_subsets(np.arange(N))
    for s in all_subsets:
        if is_subset_of(subset, s):
            super_sets.append(s)
    return super_sets


def is_subset_of(a, b):
    """returns if a is subset of b"""
    return (set(a) & set(b)) == set(a)


def subset_pairs(iterable):
    """returns all pairs of subsets of an iterable"""
    s = list(iterable)
    ps = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
    ps = [list(l) for l in ps]

    pairs = []
    pairs.append(([], []))
    for a in ps:
        for b in ps:
            if a != b:
                pair1 = (a, b)
                pair2 = (b, a)
                # make sure sets are disjoint
                if len([value for value in a if value in b]) == 0:
                    if pair1 not in pairs:
                        pairs.append(pair1)
                    if pair2 not in pairs:
                        pairs.append(pair2)
    return pairs


def intersection(list1, list2):
    """return list that is intersection of 2 lists"""
    inter = list(set(list1) & set(list2))
    inter.sort()
    return inter

def save_current_fig(figure, filename, output_dir="./out/", show=False):
    """save and/or show matplotlib figure"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print("Figure saved to: ", filepath)
    if show:
        plt.show()
    plt.close(figure)


def is_point_in_box(point, box):
    """returns if a point is within a box"""
    # point: (x, y)
    # box: (x, y, h, w)
    px, py = point
    x = box["x"]
    y = box["y"]
    h = box["h"]
    w = box["w"]
    if px >= x and px <= x + w and py >= y and py <= y + h:
        return True
    return False


def get_truncated_normal(mean=0.0, std=1.0, low=0.0, up=1.0):
    """returns truncated normal distribution with mean, std, lower and upper bound"""
    # To use
    # X = get_truncated_normal()
    # samples = X.rvs(1) to just get 1 sample
    print(low, up, (low - mean) / std, (up - mean) / std, mean, std)
    return truncnorm((low - mean) / std, (up - mean) / std, loc=mean, scale=std)


def normalize_to_range(data, min=0, max=1, mask=None):
    """returns data normalized to specified range"""

    if mask is None:
        dat_min = np.min(data)
        dat_max = np.max(data)
        normalized_data = min + (data - dat_min) * (max - min) / (
            dat_max - dat_min
        )
        return normalized_data
    else:
        masked_data = data[mask]
        dat_min = np.min(masked_data)
        dat_max = np.max(masked_data)
        normalized_mask_data = min + (masked_data- dat_min) * (max - min) / (
            dat_max - dat_min
        )
        normalized_data = np.zeros_like(data)
        normalized_data[mask] = normalized_mask_data
        return normalized_data 


def mark_bags(image, segs, bag_labels, color_neg=(255, 0, 0), color_pos=(0, 255, 0)):
    """returns image with marked pos and neg bags"""

    boundaries = image.copy()
    if image.ndim == 2:
        boundaries = np.stack((boundaries, boundaries, boundaries), axis=-1)

    pos_segs = np.where(bag_labels==1)[0] + 1
    neg_segs = np.where(bag_labels==0)[0] + 1

    neg_segmentation = np.where(np.isin(segs, pos_segs), -1, segs)
    pos_segmentation = np.where(np.isin(segs, neg_segs), -1, segs)

    boundaries = mark_boundaries(boundaries, neg_segmentation, color=color_neg)
    boundaries = mark_boundaries(boundaries, pos_segmentation, color=color_pos)

    return boundaries

def img_uint8_to_float(img):
    return img.astype(np.float32) / 255
