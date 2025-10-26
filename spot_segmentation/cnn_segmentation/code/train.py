import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
# import pandas as pd
from IPython import embed
import cv2
# from PIL import Image
import torch
# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import tifffile 
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from model import SpotModel
from make_dataset import SpotDataset
from functions import visualize_sample, visualize_predictions
import wandb

# Create an instance of the dataset
train_dataset = SpotDataset(
    image_dir='../../../data/patch_256/train/images',
    mask_dir='../../../data/patch_256/train/masks',
    transform=None,
)

val_dataset = SpotDataset(
    image_dir='../../../data/patch_256/val/images',
    mask_dir='../../../data/patch_256/val/masks',
    transform=None,
)

test_dataset = SpotDataset(
    image_dir='../../../data/patch_256/test/images',
    mask_dir='../../../data/patch_256/test/masks',
    transform=None,
)

# Make dataloaders
batch_size=4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Check an image to see all is ok
visualize_sample(train_dataset)



architecture =  'DeepLabV3plus' #'Unetplusplus' #
encoder = 'resnet34'
track_wandb = True

if track_wandb == True:
    # Start a W&B Run with wandb.init
    run = wandb.init(project=architecture)
    config = {
    "architecture": architecture,
    "encoder": encoder,
    "batch_size": batch_size 
}
    

# Some training hyperparameters
EPOCHS = 100
T_MAX = EPOCHS * len(train_dataloader)
OUT_CLASSES = 1

# Initialize model

preprocess_input = get_preprocessing_fn(encoder, pretrained='imagenet')
model = SpotModel(architecture, encoder, in_channels=3, out_classes=1, T_MAX =T_MAX, track_wandb=track_wandb)
#  Early stopping
early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=False, mode="max")

trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1, callbacks=[early_stop_callback])

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# run validation dataset
valid_metrics = trainer.validate(model, dataloaders=val_dataloader, verbose=False)
print(valid_metrics)

# run test dataset
test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
print(test_metrics)


# Mark the run as finished, and finish uploading all data
if track_wandb == True:
    run.finish()
