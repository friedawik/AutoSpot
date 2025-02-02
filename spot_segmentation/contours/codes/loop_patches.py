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

def get_mask_position(img_id, overlap=0):
    x_num = int(img_id[-8])
    y_num = int(img_id[-5])
    #print(f'x={x_num*img_size}, y={y_num*img_size}')
    x_start =max(0, x_num * img_size - overlap)
    y_start = max(0, y_num * img_size - overlap)
    x_end = min(2048, x_start + img_size - overlap)
    y_end = min(2048, y_start + img_size - overlap)

    return x_start, y_start, x_end, y_end

def paste_patch(img_id, patch_mask, fullsize_mask,overlap=0):
    x_start, y_start, x_end, y_end = get_mask_position(img_id, overlap=overlap)
    #print(f'x_start: {x_start}, y_start: {y_start}')
    fullsize_mask[x_start:x_end, y_start:y_end] = patch_mask
    return fullsize_mask


# Remove all files in fullsize folder to start from clean file
fullsize_path = "../results/sam2_fullsize"

# Get all files in the folder
files = glob.glob(os.path.join(fullsize_path, '*'))

# Delete each file
for file in files:
    if os.path.isfile(file):
        os.remove(file)


# get fullsize test image names
fullsize_folder = '../../../data/fullsize/test/images'
fullsize_masks = {}

for img_id in os.listdir(fullsize_folder):
    fullsize_masks[img_id[:-4]] = np.zeros((2048, 2048), dtype=np.uint8)
    

# Count files
patches_path = "../results/sam2_patches"
file_count = sum(1 for item in os.listdir(patches_path) if os.path.isfile(os.path.join(patches_path, item)))

# Image size hardcoded here to avoid multiple calls in loop
img_size = 256
                                         

for i, img_id in enumerate(os.listdir(patches_path)):
    print(f"processing patch {img_id}: {i+1}/{file_count}")
    patch_path = f'../results/sam2_patches/{img_id[:-4]}.png'
    

    fullsize_id = img_id[:-10]

    # Open image patch and find peaks
    patch_mask = cv2.imread(patch_path,  cv2.IMREAD_UNCHANGED)
    old_fullsize = fullsize_masks[f'{img_id[:-10]}'].copy()
    new_fullsize_mask = paste_patch(img_id, patch_mask, old_fullsize,overlap=0)

    fullsize_masks[f'{img_id[:-10]}'] = new_fullsize_mask

    #if img_id[:-4] == 'MF_MaxIP_3ch_2_000_230623_544_84_R1h_XY5_x0_y0':
   

for image in fullsize_masks:
    fullsize_path = f'../results/sam2_fullsize/{image}.png'
    cv2.imwrite(fullsize_path, fullsize_masks[image], [cv2.IMWRITE_PNG_COMPRESSION, 0])







