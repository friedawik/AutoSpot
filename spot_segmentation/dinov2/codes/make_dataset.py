from torch.utils.data import Dataset
import torch
import numpy as np
import os
import cv2



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

    def __len__(self):
        # Assuming both lists are of the same length
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get images, use image filenames to be sure same image
        img_dir = os.path.join(self.image_dir,self.image_filenames[idx])
        mask_dir = os.path.join(self.mask_dir,self.image_filenames[idx][:-4]+'_masks.png') 

        # Read image
        image = cv2.imread(img_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_np = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
   
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask_np)
            mask = torch.squeeze(mask)
            mask= mask.long()
    
        # return image, mask
        data = {'image':image, 'mask':mask}

        return data