import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tools import *
from spot_segmentation_tool import SpotSegmentor
from IPython import embed
import glob

""" 
This code predicts the masks on each patch and assembles all patches together into the full image.

"""

def get_mask_position(img_id, overlap=0):
    x_num = int(img_id[-8])
    y_num = int(img_id[-5])
    x_start =max(0, x_num * img_size - overlap)
    y_start = max(0, y_num * img_size - overlap)
    x_end = min(2048, x_start + img_size - overlap)
    y_end = min(2048, y_start + img_size - overlap)

    return x_start, y_start, x_end, y_end

def paste_patch(img_id, patch_mask, fullsize_mask,overlap=0):
    x_start, y_start, x_end, y_end = get_mask_position(img_id, overlap=overlap)
    fullsize_mask[x_start:x_end, y_start:y_end] = patch_mask
    return fullsize_mask

# data folder 
data_folder = "../../../data/patch_256/test/images"
file_count = sum(1 for item in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, item)))

# get fullsize test image names
fullsize_folder = '../../../data/fullsize/test/images'
fullsize_masks = {}

for img_id in os.listdir(fullsize_folder):
    fullsize_masks[img_id[:-4]] = np.zeros((2048, 2048), dtype=np.uint8)
    

# Remove all files in fullsize folder to start from clean file
fullsize_path = "../results/fullsize"

# Get all files in the folder
files = glob.glob(os.path.join(fullsize_path, '*'))

# Delete each file
for file in files:
    if os.path.isfile(file):
        os.remove(file)


# Image size hardcoded here to avoid multiple calls in loop
img_size = 256

# Set parameters for contours search
min_area= 3
max_area= 200
levels = 200
min_z = 100

peaks_fullsize = {}
                                         

for i, img_id in enumerate(os.listdir(data_folder)):
    print(f"processing patch {img_id}: {i+1}/{file_count}")
    patch_path = f'../results/patches/{img_id[:-4]}.png'
    fullsize_id = img_id[:-10]

    # Open image patch and find peaks
    img_path = os.path.join(data_folder,img_id)
    img = cv2.imread(img_path,  cv2.IMREAD_UNCHANGED)
    img = img[:,:,1] # check single channel data, should be somewhere

    # Get contours
    spots=SpotSegmentor(img,contour_levels=levels, min_area=min_area, max_area=max_area,min_circularity = 0.3, z_min_dist = min_z)
    patch_mask = spots.get_mask()
    mask_255 = patch_mask*255
    cv2.imwrite(patch_path, mask_255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    new_fullsize_mask = paste_patch(img_id, patch_mask, fullsize_masks[f'{img_id[:-10]}'],overlap=0)
    fullsize_masks[f'{img_id[:-10]}'] = new_fullsize_mask



for image in fullsize_masks:
    fullsize_path = f'../results/fullsize/{image}.png'
    cv2.imwrite(fullsize_path, fullsize_masks[image], [cv2.IMWRITE_PNG_COMPRESSION, 0])







