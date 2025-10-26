import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.path import Path
#import matplotlib.patches as patches
#import tifffile
import pandas as pd
from tools import *
from spot_segmentation_tool import SpotSegmentor
from IPython import embed
import glob

def pixel_comparison(mask1, mask2):
    TP = np.sum((mask1 == 255) & (mask2 == 255))
    FP = np.sum((mask1 == 0) & (mask2 == 255))
    TN = np.sum((mask1 == 0) & (mask2 == 0))
    FN = np.sum((mask1 == 255) & (mask2 == 0))
    return TP, FP, TN, FN


# Remove all files in fullsize folder to start from clean file
prediction_folder = "../results/fullsize"

# get fullsize test image names
gt_folder = '../../../data/fullsize/test/masks'


for img_id in os.listdir(prediction_folder):
    print(f'{img_id}')
    predicted_path = os.path.join(prediction_folder, img_id)
    predicted_mask = cv2.imread(predicted_path,  cv2.IMREAD_UNCHANGED)
    gt_path = os.path.join(gt_folder, img_id[:-4]+ '_masks.png')
    gt_mask = cv2.imread(gt_path,  cv2.IMREAD_UNCHANGED)

    # Calculate performance
    TP, FP, TN, FN = pixel_comparison(predicted_mask, gt_mask)

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    iou = TP / (TP + FP + FN)
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    print(f'accuracy: {accuracy}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'f1 score: {f1_score}')
    print(f'iou: {iou}')












