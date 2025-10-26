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
This code runs a pretrained SAM2 prediction with peak prompts from 'mountains' code.
Note that amg must be set to amg=True.

The following post-processing threshold were tested. Modifying them will change 
the segmentation outcome:

    pred_iou_thresh=0.75,  
    stability_score_thresh=0.65,  
    min_mask_region_area=2,  
    crop_nms_thresh=0.6

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
img_type = 'refed'
peak_dir=f'../../../peak_detection/results/results_patch/{img_type}'
img_dir=f'../../../data/patch_sam_{img_type}/test/images'
gt_dir=f'../../../data/patch_sam_{img_type}/test/masks'


# configuration
MODEL_TYPE = "vit_b"
predictor = get_sam_model(
model_type=MODEL_TYPE,
device="cuda" if torch.cuda.is_available() else "cpu"  # Will use CUDA if available, otherwise CPU
)
 

# lists to store performance metrics
iou_list = []
precision_list = []
recall_list = []
f1_list = []
acc_list = []

for file in os.listdir(peak_dir):
    print(f'processing {file}')
    img_id = file[:-4]

    # get image and gt masks
    image = imread(os.path.join(img_dir,img_id +'.png'))
    gt_mask = imread(os.path.join(gt_dir,img_id +'_masks.png'))
    gt_mask = gt_mask/255   # 1 used for positive

    # Get image dimensions
    height, width = image.shape[:2]

    # get peak coordinates
    coords_df = pd.read_csv(os.path.join(peak_dir,img_id +'.txt'))
    point_coords = coords_df[['x', 'y']].values
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
   
   # GEnerate masks
    masks = amg.generate(
    pred_iou_thresh=0.75,  # Filter masks with IoU prediction below this threshold
    stability_score_thresh=0.65,  # Filter masks with stability score below this threshold
    min_mask_region_area=2,  # Minimum mask area in pixels (filter small masks)
    crop_nms_thresh=0.6
    )

    # Make zero array for binary mask
    pred_mask = np.zeros((height, width), dtype=np.uint8)

    # Sort masks by area (largest first) to handle overlaps
    masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)

    for idx, mask_data in enumerate(masks_sorted):
        mask = mask_data['segmentation']
        # filter out too large masks that are probably wrong, should do this better way
        true_count = np.sum(mask)
        if true_count<1500: # remove large masks that cannot be correct 
        # Assign unique label to each instance
            pred_mask[mask] = 1
        else:
            print('mask probably too large')

    # calculate metrics and make plot only if image has gt masks
    if np.any(gt_mask > 0):
        precision, recall, f1_score, iou, accuracy = pixel_level_metrics(pred_mask, gt_mask)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)
        iou_list.append(iou)
        acc_list.append(accuracy)


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
        plt.axis('off')
        plt.savefig(f'../results/amg_mountains/{img_id}.png', bbox_inches='tight', pad_inches=0)
        plt.clf()

# calculate total performance
total_precision = np.mean(precision_list)
total_recall = np.mean(recall_list)
total_f1 = np.mean(f1_list)
total_iou = np.mean(iou_list)
total_acc = np.mean(acc_list)

print(f'Precision: {total_precision}\nRecall: {total_recall}\nf1_score: {total_f1}\nIOU:{total_iou}\nAccuracy: {total_acc}')
   
