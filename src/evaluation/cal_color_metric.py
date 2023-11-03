#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   cal_depth_metric.py
@Time    :   2023/08/06 21:25:53
@Author  :   Weihao Xia
@Version :   1.0
@Desc    :   None

reference:
# https://github.com/haoychen3/CD-Flow
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

def compute_stress_color(image1, image2):
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

def color_discrepancy(image1, image2, bins=64):
    """
    Compute the Color Discrepancy (CD) between two images.
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
    :return: Normalized histogram.
    """
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_images_in_dirs(dir1, dir2, bins=64):
    filenames1 = set(os.listdir(dir1))
    filenames2 = set(os.listdir(dir2))

    filenames1 = {f for f in filenames1 if f.endswith('.png')}
    filenames2 = {f for f in filenames2 if f.endswith('.png')}

    common_files = filenames1.intersection(filenames2)
    print(f'Number of common files: {len(common_files)}')
    total_discrepancy = 0
    totoal_stress = 0 

    for filename in common_files:
        image1 = cv2.imread(os.path.join(dir1, filename))
        image2 = cv2.imread(os.path.join(dir2, filename))

        cd = color_discrepancy(image1, image2, bins)
        total_discrepancy += cd
        # import pdb; pdb.set_trace()

        stress = compute_stress_color(image1, image2)
        totoal_stress += stress

    average_discrepancy = total_discrepancy / len(common_files) if common_files else 0
    average_stress = totoal_stress / len(common_files) if common_files else 0
    return average_discrepancy, average_stress

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

    avg_discrepancy, avg_stress = compare_images_in_dirs(args.recon_path, args.all_images_path, bins=64)

    # Display in Table: create a dictionary to store variable names and their corresponding values
    # print(f'({args.method}) CD: {avg_discrepancy:.2f}, STRESS: {avg_stress:.2f}')
    data = {
        "Metric": ["CD", "STRESS"],
        "Value": [avg_discrepancy, avg_stress],
    }

    df = pd.DataFrame(data)
    print(df.to_string(index=False))

    # save table to a csv file
    if args.save_csv:
        df.to_csv(f'result_color_{args.method}.csv', sep='\t', index=False, float_format='%.4f', header=True)
