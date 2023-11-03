#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   cal_depth_metric.py
@Time    :   2023/08/06 21:25:53
@Author  :   Weihao Xia 
@Version :   1.0
@Desc    :   None

reference:
https://github.com/mattpoggi/mono-uncertainty/blob/master/evaluate.py
https://github.com/nianticlabs/depth-hints/blob/master/evaluate_depth.py
'''

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

def read_pfm(path):
    """
    Read pfm file.
    Args: path (str): path to file
    Returns: tuple: (data, scale)
    """
    with open(path, "rb") as file:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale

def compute_ssim(gt, pred):
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Args:
        gt (numpy.ndarray): Ground truth image
        pred (numpy.ndarray): Predicted image

    Returns:
        float: SSIM value between the two images
    """
    return ssim(gt, pred, data_range=gt.max() - gt.min(), multichannel=True)

def compute_errors_all(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """

    # just mask out zero values
    mask_1 = (gt > 0)
    mask_2 = (pred > 0)
    mask = mask_1 & mask_2

    # apply masks
    pred = pred[mask]
    gt = gt[mask]
    
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'rmse': Root mean squared error
            'ssim': Structural similarity index
    """
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    ssim = compute_ssim(gt, pred)

    return dict(rmse=rmse, ssim=ssim)


def compute_metrics_for_folder_pairs(mode, folder1, folder2):
    # Get the list of image file names in each folder
    file_names1 = sorted(os.listdir(folder1))
    file_names2 = sorted(os.listdir(folder2))

    if mode == 'pfm':
        file_names1 = [f for f in file_names1 if f.endswith('.pfm')]
        file_names2 = [f for f in file_names2 if f.endswith('.pfm')]
    if mode == 'png':
        file_names1 = [f for f in file_names1 if f.endswith('.png')]
        file_names2 = [f for f in file_names2 if f.endswith('.png')]
    # Make sure the number of images in the two folders is the same
    if len(file_names1) != len(file_names2):
        raise ValueError("The number of images in the two folders does not match.")

    # Initialize an empty list to store the metrics for each image pair
    metrics_list = []

    # Loop through each image pair in the folders
    for filename1, filename2 in zip(file_names1, file_names2):
        # Load the images
        if mode == 'png':
            image1 = cv2.imread(os.path.join(folder1, filename1), cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(os.path.join(folder2, filename2), cv2.IMREAD_GRAYSCALE)
        if mode == 'pfm':
            image1, scale1 = read_pfm(os.path.join(folder1, filename1))
            image2, scale2 = read_pfm(os.path.join(folder2, filename2))

        # if image1.shape[0] != 425:
        #     image1 = cv2.resize(image1, (425, 425))

        # Compute the metrics for the image pair
        # metrics = compute_errors(image1, image2)
        metrics = compute_errors_all(image1, image2)

        # Append the metrics to the list
        metrics_list.append(metrics)

    return metrics_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Depth Evaluation")
    parser.add_argument(
        "--method", type=str, default='dream',
        help="name of the method",
    )
    parser.add_argument(
        "--save_csv", type=bool, default=True,
        help="save to csv file",
    )
    parser.add_argument(
        "--mode", type=str, default='png',
        help="depth pattern: png (unit8) | pfm (float32)",
    )
    parser.add_argument(
        "--recon_path", type=str, default='depth_results/reconstructed_images',
        help="path to reconstructed images",
    )
    parser.add_argument(
        "--all_images_path", type=str, default='depth_results/test_images',
        help="path to ground truth images",
    )
    args = parser.parse_args()

    metrics_list = compute_metrics_for_folder_pairs(args.mode, args.recon_path, args.all_images_path)

    # Print the average metrics for all image pairs
    print("Average metrics:")
    average_metrics = dict()
    for key in metrics_list[0].keys():
        average_metrics[key] = np.mean([metrics[key] for metrics in metrics_list])
        print(f"{key}: {average_metrics[key]}")

    # Display in Table: create a dictionary to store variable names and their corresponding values
    data = {
        "Metric": list(average_metrics.keys()),
        "Value": list(average_metrics.values()),
    }

    df = pd.DataFrame(data)
    print(df.to_string(index=False))

    # save table to a csv file
    if args.save_csv:
        df.to_csv(f'result_depth_{args.method}.csv', sep='\t', index=False, float_format='%.4f', header=True)