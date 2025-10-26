import numpy as np
import pandas as pd
from imageio import imread, imwrite
from micro_sam.util import get_sam_model
from micro_sam.instance_segmentation import AutomaticMaskGenerator
import os
from IPython import embed
from segmentation_tools import pixel_level_metrics
import matplotlib.pyplot as plt
import cv2

"""
Code to filter peaks on prominence before giving them to SAM2 as prompts.
The results are plotted and saved.
"""

def normalize_points(point_coords, width=256, height=256):
    # Normalize coordinates to [0, 1] as required by AMG
    # Note: points are in (x, y) format, need to normalize by (width, height)
    normalized_points = point_coords.copy().astype(np.float32)
    normalized_points[:, 0] = normalized_points[:, 0] / width   # normalize x
    normalized_points[:, 1] = normalized_points[:, 1] / height  # normalize y
    # AMG expects point_grids as a list of arrays (one per crop layer)
    # For simplicity, we'll use a single crop layer with our custom points
    return [normalized_points]



# data directories
img_id = 'MF_MaxIP_3ch_2_000_230623_544_84_R1h_XY5_x1_y5'
img_type = 'refed'
peak_path=f'../../../peak_detection/results/results_patch/{img_type}/{img_id}.txt'
img_path=f'../../../data/patch_sam_{img_type}/test/images/{img_id}.png'

# choose prominence limit for which peaks to include
prom_limit=300

# configuration
MODEL_TYPE = "vit_b"
predictor = get_sam_model(
model_type=MODEL_TYPE,
device=None  # Will use CUDA if available, otherwise CPU
)
 
# Get image and gt masks
image = imread(img_path)

# Get image dimensions
height, width = image.shape[:2]

# get peak coordinates
coords_df = pd.read_csv(peak_path)
filtered_df = coords_df[coords_df['prominence'] > prom_limit]
point_coords = filtered_df[['x', 'y']].values
custom_point_grids = normalize_points(point_coords)

# Create AMG instance with custom point grid
amg = AutomaticMaskGenerator(
    predictor,
    points_per_side=None,  # Must be None when using point_grids
    point_grids=custom_point_grids,
    points_per_batch=64,  # Process multiple points in parallel
    crop_n_layers=0,  # Number of crop layers (0 = no crops, just use full image)
    stability_score_offset=1.0,  # Offset for stability score calculation
)

# Initialize the image (compute embeddings)
amg.initialize(image)

# produce masks
masks = amg.generate(
pred_iou_thresh=0.85,  # Filter masks with IoU prediction below this threshold
stability_score_thresh=0.85,  # Filter masks with stability score below this threshold
min_mask_region_area=10,  # Minimum mask area in pixels (filter small masks)
crop_nms_thresh=0.8
)

# make array for binary mask
pred_mask = np.zeros((height, width), dtype=np.uint8)

# Sort masks by area (largest first) to handle overlaps
masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)

for idx, mask_data in enumerate(masks_sorted):
    mask = mask_data['segmentation']
    # filter out too large masks that are probably wrong, should do this better way
    true_count = np.sum(mask)
    if true_count<1500: # remove too large masks that are not correct
    # Assign unique label to each instance
        pred_mask[mask] = 1
    else:
        print('mask probably too large')


    # Create a plot
    fig, ax = plt.subplots(figsize=[6.4, 4.8])

    # Find contours in the predicted mask
    contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    ax.imshow(image, cmap='gray')
    # Plot each contour
    for contour in contours:
        # Reshape the contour to 2D for plotting
        contour = contour.squeeze()
        if len(contour.shape) == 2:  # Avoid empty contours
            contour =  np.vstack([contour, contour[0]])
            ax.plot(contour[:, 0], contour[:, 1], c='g', linewidth=1)  # Green contour
    ax.plot(point_coords[:, 0], point_coords[:, 1], 'o', color='red', markersize=3)  # 'o' specifies marker type
    plt.axis('off')
    plt.savefig(f'../results/diff_prompts/prompts_{str(prom_limit)}.png', bbox_inches='tight', pad_inches=0)
    plt.clf()


