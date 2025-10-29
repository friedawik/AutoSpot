from glob import glob
import os
from IPython import embed
import numpy as np
import cv2

import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components

import torch
from torch_em.util.util import get_random_colors

from micro_sam.util import get_sam_model
from micro_sam.evaluation import inference
from micro_sam.evaluation.model_comparison import _enhance_image
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data
from micro_sam.evaluation.evaluation import run_evaluation, run_evaluation_for_iterative_prompting
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from segmentation_tools import pixel_level_metrics

"""
This code predicts spot masks using a finetuned micro-sam model. To get the total performance,
the code loops through all image patches, calculates the performance and saves the segmentations
in plots for visualisation. Note that the prediction thresholds can change the predictions, we have
tested with the following values:

        center_distance_threshold = 0.60,
        boundary_distance_threshold = 0.80,
        foreground_threshold = 0.45,
        min_size = 2,


"""

def run_automatic_instance_segmentation(image, checkpoint_path, model_type="vit_b_lm", device=None):
    """Automatic Instance Segmentation (AIS) by training an additional instance decoder in SAM.

    NOTE: AIS is supported only for `µsam` models. Make sure amg is set to amg=False.

    Args:
        image: The input image.
        checkpoint_path: The path to stored checkpoints.
        model_type: The choice of the `µsam` model.
        device: The device to run the model inference.

    Returns:
        The prediction.
    """

    # Step 1: Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, # choice of the Segment Anything model
        checkpoint=checkpoint_path,  # overwrite to pass your own finetuned model.
        device=device,  # the device to run the model inference.
        amg=False,
    )

    # Step 2: Get the instance segmentation for the given image.
    prediction = automatic_instance_segmentation(
        predictor=predictor,  # the predictor for the Segment Anything model.
        segmenter=segmenter,  # the segmenter class responsible for generating predictions.
        input_path=image,
        ndim=2,

        center_distance_threshold = 0.60,
        boundary_distance_threshold = 0.80,
        foreground_threshold = 0.45,
        # foreground_smoothing = 0.75,
        # distance_smoothing = 1.6,
        min_size = 2,
    )

    return prediction

def plot_mask_contours(prediction, image_plot, result_folder, gt_id):
    # Find contours in the predicted mask
    contours, _ = cv2.findContours(prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the overlay using cv2
    cv2.drawContours(image_plot, contours, -1, (0, 255, 0), 1) # Green contours with thickness of 2
    # Plot each contour using plt.plot 
    plt.imshow(image_plot, cmap='gray')
    for contour in contours:
        # Reshape the contour to 2D for plotting to use same method as other
        contour = contour.squeeze()
        if len(contour.shape) == 2:  # Avoid empty contours
            contour =  np.vstack([contour, contour[0]])
            plt.plot(contour[:, 0], contour[:, 1], c='g', linewidth=1)
    plt.axis('off')
    plt.savefig(os.path.join(result_folder, f'{gt_id}.png'), bbox_inches='tight', pad_inches=0)
    plt.clf()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu" # the device/GPU used for training

# Set up model 
model_type = "vit_b"  
best_checkpoint = os.path.join("../models", "checkpoints", "instance", "best.pt")

# Make folder for saving plots
result_folder = os.path.join('../results', "ais")
os.makedirs(result_folder, exist_ok=True)

# Image directories
test_img_folder = '../../../data/patch_256_sam/test/images'
test_mask_folder = '../../../data/patch_256_sam/test/masks'

# Get all paths
test_mask_paths = sorted([os.path.join(test_mask_folder, f) for f in os.listdir(test_mask_folder)])

# Initiate lists to store performance metrics for each patch
iou_list = []
precision_list = []
recall_list = []
f1_list = []
acc_list = []

for gt_path in test_mask_paths:

    # get images and gt masks
    gt_mask = imageio.imread(gt_path)
    gt_id = os.path.split(gt_path)[-1][:-10]
    image_path = os.path.join(test_img_folder, gt_id+ '.png') # for sam data
    image = imageio.imread(image_path)
    image_plot = image.copy()
    print(f'Processing {gt_id}')

    # Predicted instances
    prediction = run_automatic_instance_segmentation(
        image=image, checkpoint_path=best_checkpoint, model_type=model_type, device=device
    )
    # Set all elements greater than 0 to 1 to make binary mask
    prediction[prediction > 0] = 1
    prediction=prediction.astype('uint8')
    gt_mask[gt_mask > 0] = 1
    
    # Plot and save results
    plot_mask_contours(prediction, image_plot, result_folder, gt_id )

    # Get performance metrics for this patch and save them in list
    precision, recall, f1_score, iou, balanced_accuracy = pixel_level_metrics(prediction, gt_mask)

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1_score)
    iou_list.append(iou)
    acc_list.append(balanced_accuracy)
  
# Get total performance
total_precision = np.mean(precision_list)
total_recall = np.mean(recall_list)
total_f1 = np.mean(f1_list)
total_iou = np.mean(iou_list)
total_acc = np.mean(acc_list)

print(f'Precision: {total_precision}\nRecall: {total_recall}\nf1_score: {total_f1}\nIOU:{total_iou}\nAccuracy: {total_acc}.')
 
