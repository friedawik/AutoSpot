import os
import cv2
from IPython import embed
import numpy as np

""" Script to save contours in yolo format """

# Path to data folder
data_dir = os.path.join(os.getcwd(), 'patch_640')

# Loop through all dataset folders and open each mask image. 
# Get the contours and save in a txt file following yolo format

for item in os.listdir(data_dir):
    folder_dir = os.path.join(data_dir,item)
    
    if os.path.isdir(folder_dir):
        dataset_dir = os.path.join(folder_dir, 'masks')
        for mask_image in os.listdir(dataset_dir):
            output_dir = os.path.join(folder_dir, 'labels')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{mask_image[:-10]}.txt')
            mask_dir = os.path.join(dataset_dir,mask_image)
            # Read images and extract all contours
            all_masks = cv2.imread(mask_dir)
            img_shape = all_masks.shape[0]  # can only handle squared patches
            contours, _ = cv2.findContours(all_masks[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Flatten contours and save in txt file
            with open(output_file, 'w') as f:
                for coords in contours:
                    flat_coords = coords.flatten()
                    norm_coords = flat_coords / img_shape
                    full_line =  np.insert(norm_coords, 0, 0) # add 0 for spot class
                    if len(full_line) > 5:
                     
            
                        coordinates_str = ' '.join(map(str, full_line))
                        f.write(f'{coordinates_str}\n')
         