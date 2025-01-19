from torch.utils.data import Dataset, DataLoader
import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers import MaskFormerForInstanceSegmentation
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import os
import cv2
from IPython import embed
from torchvision import transforms
import albumentations as A
import evaluate
import matplotlib.pyplot as plt
from make_dataset import SpotDataset
from dino_model import Dinov2ForSemanticSegmentation
# from dino_mask_model import Dinov2ForSemanticSegmentation

import wandb

# Start a W&B Run with wandb.init
run = wandb.init(project="Dinov2")

def show_img(image):
    image =image.squeeze().cpu()
    if image.shape[0] == 3:
      image=image.permute(1, 2, 0)
    
    plt.imshow(image)
    
    plt.savefig('test.png')

# Define transformations 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((448, 448))
])


# Create datasets
train_dataset = SpotDataset(
    image_dir='../patch_256_8bit/train/images',
    mask_dir='../patch_256_8bit/train/masks',
    transform=transform,
)

val_dataset = SpotDataset(
    image_dir='../patch_256_8bit/val/images',
    mask_dir='../patch_256_8bit/val/masks',
    transform=transform,
)

test_dataset = SpotDataset(
    image_dir='../patch_256_8bit/test/images',
    mask_dir='../patch_256_8bit/test/masks',
    transform=transform,
)

# Make labels
id2label = {0:'background', 1:'spot'}

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    return {'image': images, 'mask': masks}

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Test the dataloader
batch = next(iter(train_dataloader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)

# Initialize model
model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", id2label=id2label, num_labels=len(id2label))


# Freeze backbone
for name, param in model.named_parameters():
  if name.startswith("dinov2"):
    param.requires_grad = False


outputs = model(pixel_values=batch["image"], labels=batch["mask"])
print(outputs.logits.shape)
print(outputs.loss)


metric = evaluate.load("mean_iou")

learning_rate = 5e-4
epochs = 21
save_epoch = 5

optimizer = AdamW(model.parameters(), lr=learning_rate)

# put model on GPU 
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# put model in training mode
model.train()

for epoch in range(epochs):
  print("Epoch:", epoch)
  for idx, batch in enumerate(tqdm(train_dataloader)):
      pixel_values = batch["image"].to(device)
      labels = batch["mask"].to(device)

      # forward pass
      outputs = model(pixel_values, labels=labels)
      loss = outputs.loss
      loss.backward()
      optimizer.step()

      # zero the parameter gradients
      optimizer.zero_grad()

            # evaluate
      with torch.no_grad():
        predicted = outputs.logits.argmax(dim=1)

        # note that the metric expects predictions + labels as numpy arrays
        metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

      # print loss and metrics every 100 batches
      if idx % len(train_dataloader) == 0:
        metrics = metric.compute(num_labels=len(id2label),
                                ignore_index=-100,
                                reduce_labels=False,
        )

        print("Loss:", loss.item())
        print("Mean_iou:", metrics["mean_iou"])
        print("Mean accuracy:", metrics["mean_accuracy"])
        wandb.log({"train acc": metrics["mean_accuracy"], " train loss": loss.item(),"train mean_iou:": metrics["mean_iou"]})

  model.eval()
  for idx, batch in enumerate(tqdm(val_dataloader)):
      pixel_values = batch["image"].to(device)
      labels = batch["mask"].to(device)

      # forward pass
      # outputs = model(pixel_values, labels=labels)
      outputs = model(pixel_values)
      # loss = outputs.loss

      # loss.backward()
      # optimizer.step()

      # # zero the parameter gradients
      # optimizer.zero_grad()

            # evaluate
      with torch.no_grad():
        predicted = outputs.logits.argmax(dim=1)

        # note that the metric expects predictions + labels as numpy arrays
        metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

      # print loss and metrics every 100 batches
      if idx % len(val_dataloader) == 0:

        metrics = metric.compute(num_labels=len(id2label),
                                # ignore_index=0,
                                ignore_index=-100,
                                reduce_labels=False,
        )


        print("Val mean_iou:", metrics["mean_iou"])
        print("Val mean accuracy:", metrics["mean_accuracy"])

        wandb.log({"val acc": metrics["mean_accuracy"], "val mean_iou:": metrics["mean_iou"]})
   
  # if epoch//save_epoch==0:
  torch.save(model, f"dinomodels/epoch{epoch}.pt")

# Save model inputs and hyperparameters in a wandb.config object
config = run.config
config.learning_rate = learning_rate

# Mark the run as finished, and finish uploading all data
run.finish()