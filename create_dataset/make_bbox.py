import os
import cv2
from IPython import embed
import numpy as np

# Path to data folder
data_dir = os.path.join(os.getcwd(), 'patch_256_detection')

# Loop through all dataset folders and open each mask image. 
# Get the contours and save bbox in a txt file following yolo format

for item in os.listdir(data_dir):
    folder_dir = os.path.join(data_dir,item)
    
    if os.path.isdir(folder_dir):
        dataset_dir = os.path.join(folder_dir, 'masks')
        for mask_image in os.listdir(dataset_dir):
            output_dir = os.path.join(folder_dir, 'labels_bbox')
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
                            x, y, w, h = cv2.boundingRect(coords)

                            # Convert to YOLO format (normalized)
                            img_h, img_w = img_shape,img_shape
                            x_center = (x + w / 2) / img_w
                            y_center = (y + h / 2) / img_h
                            width = w / img_w
                            height = h / img_h

                            # Write to file (assuming class 0)
                            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            