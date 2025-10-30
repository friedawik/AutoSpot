import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        # List all images in the provided directory
        self.images = [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        print(image)
        if self.transform:
            image = self.transform(image)
        
        return image  # Only return the image; return label if you have it

# Define the transformation to apply to the dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Use the custom dataset to load images from your directory
directory = '../../../../data/patch_256/train/images'  # Replace with your path
dataset = CustomImageDataset(directory=directory, transform=transform)

# Initialize variables to hold the sum and count of pixels
mean = torch.zeros(3)  # Assuming 3 channels for RGB
std = torch.zeros(3)
total_images = len(dataset)

# Calculate mean and std
for img in tqdm(dataset):
    print(img)
    # Calculate mean and std for each image channel
    mean += img.mean(dim=[1, 2])  # img shape is [C, H, W]
    std += img.std(dim=[1, 2])    # img shape is [C, H, W]

# Average the values to get the overall mean and std
mean /= total_images
std /= total_images

print("Mean:", mean)
print("Std:", std)
