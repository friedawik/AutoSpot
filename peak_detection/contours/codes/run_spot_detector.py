import cv2
import os
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.path import Path
#import matplotlib.patches as patches
#import tifffile
#import pandas as pd
from tools import *
from spot_detection_tool import SpotDetector
from IPython import embed
import glob

# data folder 
data_folder = "../../../data/patch_256/test/images"
file_count = sum(1 for item in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, item)))

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
min_z = 50

peaks_fullsize = {}

for i, img_id in enumerate(os.listdir(data_folder)):
    print(f"processing patch {img_id}: {i}/{file_count}")
    fullsize_id = img_id[:-10]
    # Get x,y patch position in fullsize image
    y_num = int(img_id[-8])
    x_num = int(img_id[-5])

    # Open image patch and find peaks
    img_path = os.path.join(data_folder,img_id)
    #img = cv2.imread(img_path)
    img = tiff_to_array(img_path)
    img = img[:,:,1] # check single channel data, should be somewhere

    spots=SpotDetector(img,contour_levels=levels, min_area=min_area, max_area=max_area,min_circularity = 0.3, z_min_dist = min_z)
    peak_dict = spots.peak_dict
    if peak_dict:
        peaks_patch = {}
        start_ind = len(peaks_fullsize) # just to start at the right index
        for key, value in peak_dict.items(): 
            peaks_fullsize[key+start_ind] = value
            peaks_patch[key] = value
        
        # Save patch coordinates and base level to txt file with comma separation
        patch_file = f'../results/patches/{img_id[:-4]}.txt'
        with open(patch_file, 'w') as file:
            for key, value in peaks_patch.items():
                x = int(value["centroid"][0])
                y = int(value["centroid"][1])
                centroid_str = (f"{x},{y},{value["base_level"]},{value["z_dist"]}")
                # Write the array to the file, adding a newline character
                file.write(f"{centroid_str}\n")

        # Save fullsize coordinates and base level to txt file with comma separation
        fullsize_file = f'../results/fullsize/{fullsize_id}.txt'
        with open(fullsize_file, 'a') as file:
            for key, value in peaks_patch.items():
                x = int(value["centroid"][0] + img_size * x_num)
                y = int(value["centroid"][1] + img_size * y_num)
                centroid_str = (f"{x},{y},{value["base_level"]},{value["z_dist"]}")
                # Write the array to the file, adding a newline character
                file.write(f"{centroid_str}\n")



