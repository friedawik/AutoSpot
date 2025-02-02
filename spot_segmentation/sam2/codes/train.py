import os
import pandas as pd
import cv2
import torch
import torch.nn.utils
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
from scipy.ndimage import label
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from IPython import embed
from make_dataset import SpotDataset
from tqdm.auto import tqdm

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define transformations 
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((448, 448))
])

# Get data folder
current_dir = os.getcwd()
data_dir = os.path.abspath(os.path.join(current_dir, *(['..'] * 3), 'data/patch_256_8bit'))

# Create datasets
train_dataset = SpotDataset(
    image_dir= os.path.join(data_dir, 'train/images'),
    mask_dir=os.path.join(data_dir, 'train/masks'),
    transform=transform,
)

val_dataset = SpotDataset(
    image_dir=os.path.join(data_dir, 'val/images'),
    mask_dir=os.path.join(data_dir, 'val/masks'),
    transform=transform,
)

test_dataset = SpotDataset(
    image_dir=os.path.join(data_dir, 'test/images'),
    mask_dir=os.path.join(data_dir, 'test/masks'),
    transform=transform,
)

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    return {'image': images, 'mask': masks}

batch_size=4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)



# Set up sam2

sam2_checkpoint = "sam2_hiera_tiny.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
model_cfg = "sam2_hiera_t.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)


# Train mask decoder.
predictor.model.sam_mask_decoder.train(True)

# Train prompt encoder.
predictor.model.sam_prompt_encoder.train(True)

# Configure optimizer.
optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=0.001,weight_decay=1e-4) #1e-5, weight_decay = 4e-5

# # Mix precision.
scaler = torch.amp.GradScaler()


# No. of steps to train the model.
NO_OF_STEPS = 2000 # @param 

# Fine-tuned model name.
FINE_TUNED_MODEL_NAME = "fine_tuned_sam2"


# Initialize scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2) # 500 , 250, gamma = 0.1
accumulation_steps = 4  # Number of steps to accumulate gradients before updating

# Initialize mean_iou
mean_iou = 0

epochs = 2

for epoch in range(epochs):
    print("Epoch:", epoch)
    for idx, batch in enumerate(tqdm(train_dataloader)):
        image = batch["image"].to(device)
        masks = batch["mask"].to(device)

        predictor.set_image(image)
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
           points=None, boxes=None, masks=None,
        )

        torch.cuda.empty_cache()  # Clear cache before operation
    
        batched_mode = image.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

        # print(predictor._features["image_embed"][-1].unsqueeze(0).shape )
        # print(predictor.model.sam_prompt_encoder.get_dense_pe().shape)
        # print(sparse_embeddings.shape)
        # print(dense_embeddings.shape)
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
        
        
        gt_mask = torch.tensor(masks.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05

        # Apply gradient accumulation
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

        if step % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            predictor.model.zero_grad()

            # Update scheduler
            scheduler.step()

        if step % 500 == 0:
            FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_" + str(step) + ".torch"
            torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)

        # if step == 1:
        #     mean_iou = 0
 
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

        if step % 20 == 0:
            print("Step " + str(step) + ":\t", "Accuracy (IoU) = ", mean_iou)










