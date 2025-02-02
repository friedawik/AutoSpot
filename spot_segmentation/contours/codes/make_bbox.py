import cv2
import numpy as np
import csv
from IPython import embed
import os

def extract_bboxes(binary_image):
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((y, x, y+h, x+w))  # (min_row, min_col, max_row, max_col)
    
    return bboxes

# Open the binary image
patch_folder = '../results/patches'

for i, img_id in enumerate(os.listdir(patch_folder)):
    patch_path = os.path.join(patch_folder,img_id)
    binary_image = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the image is binary (0 and 255)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # Extract bounding boxes
    bboxes = extract_bboxes(binary_image)

    # Save bounding boxes to CSV
    csv_path = f'../results/bbox_patches/{img_id[:-4]}.csv'

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['min_row', 'min_col', 'max_row', 'max_col'])
        for bbox in bboxes:
            writer.writerow(bbox)

    #print(f"Bounding boxes saved to {csv_path}")

