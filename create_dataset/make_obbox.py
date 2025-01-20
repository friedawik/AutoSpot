import os
import cv2
from IPython import embed
import numpy as np
from pathlib import Path

def get_oriented_bbox(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box

def normalize_coordinates(box, img_shape):
    h, w = img_shape,img_shape
    return [[x/w, y/h] for x, y in box]

def process_masks(mask_folder, output_folder):
    mask_folder = Path(mask_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

# Path to data folder
data_dir = os.path.join(os.getcwd(), 'patch_256_obb')

# Loop through all dataset folders and open each mask image. 
# Get the contours and save bbox in a txt file following yolo format

for item in os.listdir(data_dir):
    folder_dir = os.path.join(data_dir,item)
    
    if os.path.isdir(folder_dir):
        dataset_dir = os.path.join(folder_dir, 'masks')
        for mask_image in os.listdir(dataset_dir):
            output_dir = os.path.join(folder_dir, 'labels')
            os.makedirs(output_dir, exist_ok=True)
            
            mask_dir = os.path.join(dataset_dir,mask_image)
            # Read images and extract all contours
            all_masks = cv2.imread(mask_dir)
            img_shape = all_masks.shape[0]  # can only handle squared patches
            contours, _ = cv2.findContours(all_masks[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                output_file = os.path.join(output_dir, f'{mask_image[:-10]}.txt')
                # Flatten contours and save in txt file
                with open(output_file, 'w') as f:
                    for coords in contours:
                        # Get bounding box coordinates
                        box = get_oriented_bbox(coords)
                        normalized_box = normalize_coordinates(box, img_shape)
                        
                        # Ensure clockwise order starting from top-left
                        normalized_box = sorted(normalized_box, key=lambda p: (p[1], p[0]))
                        top_two = sorted(normalized_box[:2], key=lambda p: p[0])
                        bottom_two = sorted(normalized_box[2:], key=lambda p: -p[0])
                        normalized_box = top_two + bottom_two

                        # Write to file (assuming class 0)
                        box_str = ' '.join(f'{x:.6f} {y:.6f}' for x, y in normalized_box)
                        f.write(f'0 {box_str}\n')