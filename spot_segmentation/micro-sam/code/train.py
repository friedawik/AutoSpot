import warnings
warnings.filterwarnings("ignore")

from glob import glob
import os
from IPython.display import FileLink

import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components

import torch

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.util.util import get_random_colors

import micro_sam.training as sam_training
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

"""
Code to train micro-sam model with segmentation decoder. The code is adapted
from https://github.com/computational-cell-analytics/micro-sam.
"""

train_img_folder = '../../../data/patch_256_sam/train/images'
train_mask_folder = '../../../data/patch_256_sam/train/masks_instances'
val_img_folder = '../../../data/patch_256_sam/val/images'
val_mask_folder = '../../../data/patch_256_sam/val/masks_instances'



# Load images from multiple files in folder via pattern (here: all tif files)
raw_key, label_key = "*.png", "*.png"

batch_size = 1  # the training batch size
patch_shape = (1, 256, 256)  # the size of patches for training

# Train an additional convolutional decoder for end-to-end automatic instance segmentation
train_instance_segmentation = True

sampler = MinInstanceSampler(min_size=25)  

train_loader = sam_training.default_sam_loader(
    raw_paths=train_img_folder,
    raw_key=raw_key,
    label_paths=train_mask_folder,
    label_key=label_key,
    with_segmentation_decoder=train_instance_segmentation,
    patch_shape=patch_shape,
    batch_size=batch_size,
    is_seg_dataset=True,
    rois=None,
    shuffle=True,
    raw_transform=sam_training.identity,
    sampler=sampler,
)

val_loader = sam_training.default_sam_loader(
    raw_paths=val_img_folder,
    raw_key=raw_key,
    label_paths=val_mask_folder,
    label_key=label_key,
    with_segmentation_decoder=train_instance_segmentation,
    patch_shape=patch_shape,
    batch_size=batch_size,
    is_seg_dataset=True,
    rois=None,
    shuffle=True,
    raw_transform=sam_training.identity,
    sampler=sampler,
)

# All hyperparameters for training.
n_objects_per_batch = 25  # the number of objects per batch that will be sampled
device = "cuda" if torch.cuda.is_available() else "cpu" # the device/GPU used for training
n_epochs = 50  # how long we train (in epochs)

# The model_type determines which base model is used to initialize the weights that are finetuned.
model_type = "vit_b"


root_dir = '/global/D1/homes/frieda/tools/micro-sam'

# Run training
sam_training.train_sam(
    name='ais' ,
    save_root=os.path.join(root_dir, "models"),
    model_type=model_type,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=n_epochs,
    n_objects_per_batch=n_objects_per_batch,
    with_segmentation_decoder=train_instance_segmentation,
    device=device,
    early_stopping=10,
    verify_n_labels_in_loader = 100,
    freeze='image_encoder'
)