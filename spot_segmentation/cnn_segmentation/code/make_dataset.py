import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import pandas as pd
from IPython import embed
import cv2
# from PIL import Image
import torch
from torch.utils.data import Dataset
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from functions import tiff_to_array

# Define the custom dataset class
class SpotDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir: List to folder with images.
            mask_dir: List to folder with masks.
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)
        self.mask_filenames = os.listdir(mask_dir)

    def __len__(self):
        # Assuming both lists are of the same length
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

    
        img_dir = os.path.join(self.image_dir,self.image_filenames[idx])
        mask_dir = os.path.join(self.mask_dir,self.image_filenames[idx][:-4]+'_masks.png')
        # mask_dir = os.path.join(self.mask_dir,self.mask_filenames[idx])

        # Read image
        # image = cv2.imread(img_dir)
        image = tiff_to_array(img_dir)
        
        image = image.transpose(2, 0, 1) # Use if want to use all three channels for segmentation
        # Extract the second channel (index 1) and stack it to create 3 channel input
        # second_channel = image[:, :, 1]
        # new_image_np = np.stack((second_channel, second_channel, second_channel), axis=-1)
        # image = new_image_np.transpose(2, 0, 1)

        # Read mask image and convert to max = 1 instead of max = 255
        mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
        mask = mask // 255
        mask = np.expand_dims(mask, axis=0)  # Now shape is (1024, 1024, 1)
        data = {'image':image, 'mask':mask}

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # return image, mask
  
        return data

# Define transformations (example: converting to tensor and normalizing)
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # For images
# ])




