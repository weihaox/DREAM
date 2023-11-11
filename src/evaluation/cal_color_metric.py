#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   cal_depth_metric.py
@Time    :   2023/08/06 21:25:53
@Author  :   Weihao Xia
@Version :   1.0
@Desc    :   None
"""

import cv2
import os
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from PIL import Image

def compute_stress(de, dv):
    """
    Compute the Stress (STRESS) between two images.
    :param de: First image.
    :param dv: Second image.
    :return: STRESS between the two images.
    """
    fcv = np.sum(de*de)/np.sum(de*dv)
    STRESS = 100*sqrt(np.sum((de-fcv*dv)*(de-fcv*dv))/(fcv*fcv*np.sum(dv*dv)))
    return STRESS

def get_cond_color_v1(cond_image, mask_size=64):
    H, W = cond_image.size
    cond_image = cond_image.resize((W // mask_size, H // mask_size), Image.BICUBIC)
    color = cond_image.resize((H, W), Image.NEAREST)
    return color

def get_cond_color(cond_image, mask_size=64):
    H, W = cond_image.shape[:2]  # Get the height and width of the input image
    # Resize the conditional image using OpenCV
    color = cv2.resize(cond_image, (W // mask_size, H // mask_size), interpolation=cv2.INTER_CUBIC)
    # Resize the image back to the original size using nearest-neighbor interpolation
    # color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
    return color

def compute_stress(image1, image2):
    """
    Compute the Stress (STRESS) between two images.
    :param gt: First image.
    :param pred: Second image.
    :return: STRESS between the two images.
    """
    # get the color of the ground truth and prediction
    gt = get_cond_color(image1)
    pred = get_cond_color(image2)

    # convert the ground truth and prediction to a list
    gt = gt.tolist()
    pred = pred.tolist()

    # convert the ground truth and prediction to a numpy array
    gt = np.array(gt)
    pred = np.array(pred)

    STRESS = compute_stress(gt, pred)

    return STRESS

def compute_color_discrepancy(image1, image2, bins=64):
    """
    Compute the Color Discrepancy (CD) between two images.
    reference: https://github.com/haoychen3/CD-Flow
    :param image1: First image.
    :param image2: Second image.
    :param bins: Number of bins for histogram.
    :return: CD between the two images.
    """
    hist1 = compute_histogram(image1, bins)
    hist2 = compute_histogram(image2, bins)

    return np.sum(np.abs(hist1 - hist2))

def compute_histogram(image, bins):
    """
    Compute the histogram of an image.
    :param image: Input image.
    :param bins: Number of bins for histogram.
    #:param norm_type: Type of normalization (NORM_INF|NORM_L1|NORM_L2|NORM_MINMAX).
    :return: Normalized histogram.
    """
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
    # hist = cv2.normalize(hist, hist).flatten()
    # hist = hist / hist.sum() # same as using cv2.NORM_L1 if all positive
    return hist

def compute_hellinger_distance(p, q):
    """
    Compute the Hellinger distance between two vectors. 
    Hellinger distance is a metric to quantify the similarity between two probability distributions.
    Distance will be a number between <0,1>, where 0 is minimum distance (maximum similarity).
    KL divergence is not symmetric, so we use Hellinger distance instead.
    Parameters p, q: The probability distributions represented as numpy arrays.
    Returns: The Hellinger distance between the two probability distributions.
    """
    # n = len(p)
    # sum = 0.0
    # for i in range(n):
    #     sum += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    # result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

def compute_color_relevance(image1, image2, bins=32):
    """
    Compute the Color Relevance (CR) between two images based on the Hellinger distance.
    TODO: support different color spaces.
    reference: https://github.com/tody411/ColorHistogram
    :param image1: First image.
    :param image2: Second image.
    :param bins: Number of bins for histogram.
    #:param color_space: Color space for histogram (LAB|RGB|HSV).
    :return: CR between the two images.
    """
    hist1 = compute_histogram(image1, bins)
    hist2 = compute_histogram(image2, bins)
    return compute_hellinger_distance(hist1, hist2)

def compare_images_in_dirs(dir1, dir2, bins=64):
    filenames1 = set(os.listdir(dir1))
    filenames2 = set(os.listdir(dir2))

    filenames1 = {f for f in filenames1 if f.endswith('.png')}
    filenames2 = {f for f in filenames2 if f.endswith('.png')}
    assert len(filenames1) == len(filenames2), 'Number of images in two directories must be the same.'

    common_files = filenames1.intersection(filenames2)
    print(f'Number of common files: {len(common_files)}')
    total_cd = 0
    total_cr = 0
    total_stress = 0 

    for filename in common_files:
        image1 = cv2.imread(os.path.join(dir1, filename))
        image2 = cv2.imread(os.path.join(dir2, filename))

        cd = compute_color_discrepancy(image1, image2, bins)
        total_cd += cd

        cr = compute_color_relevance(image1, image2, bins)
        total_cr += cr

        stress = 0 #compute_stress(image1, image2)
        total_stress += stress

    average_cd = total_cd / len(common_files) if common_files else 0
    average_cr = total_cr / len(common_files) if common_files else 0
    average_stress = total_stress / len(common_files) if common_files else 0
    return average_cd, average_cr, average_stress

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Color Evaluation")
    parser.add_argument(
        "--method", type=str, default='dream',
        help="name of the method",
    )
    parser.add_argument(
        "--save_csv", type=bool, default=True,
        help="save to csv file",
    )
    parser.add_argument(
        "--recon_path", type=str, default='reconstructed_images',
        help="path to reconstructed images",
    )
    parser.add_argument(
        "--all_images_path", type=str, default='test_images',
        help="path to ground truth images",
    )
    args = parser.parse_args()

    avg_cd, avg_cr, avg_stress = compare_images_in_dirs(args.recon_path, args.all_images_path, bins=64)

    # Display in Table: create a dictionary to store variable names and their corresponding values
    # print(f'({args.method}) CD: {avg_cd:.2f}, CR: {avg_cr:.2f}, STRESS: {avg_stress:.2f}')
    data = {
        "Metric": ["CD", "CR", "STRESS"],
        "Value": [avg_cd, avg_cr, avg_stress],
    }

    df = pd.DataFrame(data)
    print(df.to_string(index=False))

    # save table to a csv file
    if args.save_csv:
        df.to_csv(f'result_color_{args.method}.csv', sep='\t', index=False, float_format='%.4f', header=True)
