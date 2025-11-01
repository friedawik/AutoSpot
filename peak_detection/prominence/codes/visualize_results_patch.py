import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from IPython import embed

"""
This code visualizes the results obtained from the 'mountains' algorithm applied to the 256*256 pixel patches.

Args:
    image_id (str): The identifier for the image to be visualized, provided as a command-line argument.
    min_elevation (int): The minimum elevation threshold used for visualization, specified as a command-line argument.
    min_prominence (int): The minimum prominence value for features to be included in the visualization, also specified as a command-line argument.
"""

def load_image(image_id, image_type):
    """Load an image file."""
    if image_type == 'georef':
        return cv2.imread(f'../data/images_georef/{image_id}_georef.tif', cv2.IMREAD_UNCHANGED)
    elif image_type == 'mask':
        return cv2.imread(f'../../data/patch_256/test/masks/{image_id}_masks.png', cv2.IMREAD_UNCHANGED)
    elif image_type == 'gt':
        img_temp = cv2.imread(f'../../spot_segmentation/yolo/gt_plots/{image_id}.png', cv2.IMREAD_UNCHANGED)
        # image_rgb = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)
        return img

    else:
        raise ValueError("Invalid image type")

def load_results(image_id, elevation_min, prominence_max):
    """Load and filter results from a CSV file."""
    file_path = f'../results/results_patch/{image_id}.txt'
    df = pd.read_csv(file_path)
    # Remove bad peaks on edges
    df = df[df['x']<256]
    df = df[df['y']<256]   
    df[df['elevation'] >= int(elevation_min)]
    df[df['prominence'] >= int(prominence_max)] 
    return df

def plot_results(img, masks, df, image_id):
    """Create and save plots of the results."""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), layout="constrained")
    for ax in axes:
        ax.axis('off')  # This will turn off the axes
    max_val = img.max() * 0.9
    
    # Plot original image with points
    im1 = axes[1].imshow(img, cmap='gray', vmax=max_val)
    for _, row in df.iterrows():
        axes[1].plot(row['x'], row['y'], marker='.', markersize=6, c='r')
    axes[1].set_title('Peak Detection')
    
    # Plot masks
    im2 = axes[2].imshow(masks)
    axes[2].set_title('Ground Truth')
    
    # Plot original image again
    im3 = axes[0].imshow(img, cmap='gray', vmax=max_val)
    axes[0].set_title('Original Image')
    
    # Add colorbar
    fig.colorbar(im3, ax=axes.ravel().tolist(), location='right')
    
    # Save the figure
    plt.savefig(f'../plots/{image_id}.png')

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <image_id> <elevation_min> <prominence_min>")
        sys.exit(1)

    image_id = sys.argv[1]
    elevation_min = sys.argv[2]
    prominence_min = sys.argv[3]

    # Load images
    img = load_image(image_id, 'georef')
    masks = load_image(image_id, 'mask')
    gt = load_image(image_id, 'gt')

    # Load and filter results
    df = load_results(image_id, elevation_min, prominence_max)

    # Create and save plots
    plot_results(img, gt, df, image_id)

if __name__ == "__main__":
    main()