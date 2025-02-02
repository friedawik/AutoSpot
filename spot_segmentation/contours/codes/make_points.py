import cv2
import numpy as np
import csv
from IPython import embed
import os
import random
import matplotlib.pyplot as plt

def extract_bboxes(mask, img):
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []

    for contour in contours:
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        mask_box = mask[ y:y+h, x:x+w]
        img_box = img[ y:y+h, x:x+w]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img_box)
        zero_pixels = np.where(mask_box == 0)

        if len(zero_pixels[0])<2:
            add = 5
            x_start = min(0, x - add)
            y_start = min(0,y - add)
            x_end = max(255, x+w+add)
            y_end = max(255, y+h + add)
       
            new_mask_box = mask[ y_start:y_end, x_start:x_end]
            zero_pixels = np.where(new_mask_box == 0)
      


        # Get positive point
        pos_x = max_loc[0]+x
        pos_y = max_loc[1]+y

        random_int = random.randint(0, len(zero_pixels)-1)
        
        try:
            # Get negative point
            neg_x = zero_pixels[0][random_int] + x
            neg_y = zero_pixels[1][random_int] + y

        except:
            print(i)
            print(zero_pixels)
            embed()
        
        bboxes.append((y, x, y+h, x+w, pos_x, pos_y, neg_x, neg_y))  # (min_row, min_col, max_row, max_col)
    
    return bboxes

# Open the binary image
patch_folder = '../results/patches'
data_folder = '../../../data/patch_256_8bit/test/images'

for i, img_id in enumerate(os.listdir(patch_folder)):
    patch_path = os.path.join(patch_folder,img_id)
    mask = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
    img_path = os.path.join(data_folder,img_id)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the image is binary (0 and 255)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Extract bounding boxes
    bboxes = extract_bboxes(mask, img)

    # Save bounding boxes to CSV
    csv_path = f'../results/points_patches/{img_id[:-4]}.csv'

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['min_row', 'min_col', 'max_row', 'max_col', 'pos_x','pos_y', 'neg_x','neg_y'])
        for bbox in bboxes:
            writer.writerow(bbox)

    #print(f"Bounding boxes saved to {csv_path}")

