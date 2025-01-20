import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from IPython import embed


def plot_masks(image_path, label_path, mask_path):
    # Load the image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Read the label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Create a blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Process each line
    for line in lines:
        # Split the line into coordinates
        values = list(map(float, line.strip().split()))

        # Skip the first value (class) and use the rest as coordinates
        coords = values[1:]
        
        # Convert normalized coordinates to pixel coordinates
        pixel_coords = [(int(coords[i] * width), int(coords[i+1] * height)) 
                        for i in range(0, len(coords), 2)]
        
        # Draw the polygon on the mask
        cv2.fillPoly(mask, [np.array(pixel_coords)], 255)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)


    # Display the result
    plt.subplot(1, 3, 2)
    plt.imshow(masked_image)
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 3)
    plt.imshow(mask)


    plt.savefig( 'test.png')

    # lets look at some samples
  





current_dir = os.getcwd()
image_dir = os.path.join(current_dir,'patch_640/train/images')
labels_dir = 'train/labels'

for image in os.listdir(image_dir):
    img_dir = os.path.join(current_dir,'patch_640/train/images', image)
    mask_dir = os.path.join(current_dir,'patch_640/train/masks', image[:-4] + '_masks.png')
    label_dir = os.path.join(current_dir,'patch_640/train/labels', image[:-4] + '.txt')
    # Usage
    plot_masks(img_dir, label_dir, mask_dir)
    embed()
