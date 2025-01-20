import os
import pandas as pd
import cv2
import torch
import torch.nn.utils
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from scipy.ndimage import label
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from IPython import embed

# Path to the chest-ct-segmentation dataset folder

data_dir = "finetuning/"
images_dir = os.path.join(data_dir, "images")
masks_dir = os.path.join(data_dir, "masks")

# Load the train.csv file
image_dict = {'image_id':[], 'mask_id':[]}
for filename in os.listdir(masks_dir):
   image_dict['image_id'].append(filename[:-10] + '.tiff')
   image_dict["mask_id"].append(filename)


#train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

image_df = pd.DataFrame.from_dict(image_dict)
# Split the data into two halves: one for training and one for testing
train_df, test_df = train_test_split(image_df, test_size=0.2, random_state=42)


# Prepare the training data list
train_data = []
for index, row in train_df.iterrows():
   image_name = row['image_id']
   mask_name = row['mask_id']

   # Append image and corresponding mask paths
   train_data.append({
       "image": os.path.join(images_dir, image_name),
       "annotation": os.path.join(masks_dir, mask_name)
   })

# Prepare the testing data list (if needed for inference or evaluation later)
test_data = []
for index, row in test_df.iterrows():
   image_name = row['image_id']
   mask_name = row['mask_id']

   # Append image and corresponding mask paths
   test_data.append({
       "image": os.path.join(images_dir, image_name),
       "annotation": os.path.join(masks_dir, mask_name)
   })

def read_batch(data, visualize_data=False):
   
    # Select a random entry
    ent = data[np.random.randint(len(data))]
 
    # Get full paths
    #img = cv2.imread(ent["image"])[..., ::-1]  # Convert BGR to RGB

    img_16bit = cv2.imread(ent["image"], cv2.IMREAD_UNCHANGED)
    img_16bit = img_16bit - img_16bit.min()
    img_8bit = cv2.convertScaleAbs(img_16bit, alpha=(255.0/img_16bit.max()))
    img_8bit = np.expand_dims(img_8bit, axis=-1)
    img_8bit = np.repeat(img_8bit, 3, axis=-1)
   
    #img_8bit = torch.tensor(img_8bit, dtype=torch.float32).unsqueeze(0) / 255.0  # Add channel dimension and scale to [0, 1]

# Transform the grayscale image
# Replicate the single channel across 3 channels
    img = img_8bit
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # Read annotation as grayscale
    
    # test smaller images
    chunk_size = 80  # Example chunk size
    chunk_max = int(len(img[0,:,:]) / chunk_size)
    # chunk_max = int(len(img) / chunk_size)
    chunk_x = np.random.randint(chunk_max) * chunk_size
    chunk_y = np.random.randint(chunk_max) * chunk_size
    img_chunk = img[chunk_x:chunk_x+ chunk_size, chunk_y :chunk_y + chunk_size,:]
    mask_chunk = ann_map[chunk_x : chunk_x+ chunk_size, chunk_y : chunk_y+ chunk_size]

    # img = cv2.cvtColor(img_chunk, cv2.COLOR_GRAY2RGB)
    img = img_chunk
    ann_map = mask_chunk

    if img is None or ann_map is None:
        print(f"Error: Could not read image or mask from path {ent['image']} or {ent['annotation']}")
        return None, None, None, 0

    ann_map, num_features = label(ann_map)
    # print(num_features)
   # Resize image and mask, need to check if ok!!
    # r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])  # Scaling factor
    # img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    # ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    ### Continuation of read_batch() ###

    # Initialize a single binary mask
    binary_mask = np.zeros_like(ann_map, dtype=np.uint16)
    points = []

    # Get binary masks and combine them into a single mask
    inds = np.unique(ann_map)[1:]  # Skip the background (index 0)
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint16)  # Create binary mask for each unique index
        binary_mask = np.maximum(binary_mask, mask)  # Combine with the existing binary mask
        # Erode the combined binary mask to avoid boundary points
        eroded_mask = cv2.erode(mask, np.ones((1, 1), np.uint16), iterations=1)
        # Get all coordinates inside the eroded mask and choose a random point
        coords = np.argwhere(eroded_mask == 1)
        if len(coords) > 0:
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([yx[1], yx[0]])
      

    points = np.array(points)

    if visualize_data:
        # Plotting the images and points
        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(img)
        plt.axis('off')

        # Segmentation Mask (binary_mask)
        plt.subplot(1, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')

        # Mask with Points in Different Colors
        plt.subplot(1, 3, 3)
        plt.title('Binarized Mask with Points')
        plt.imshow(binary_mask, cmap='gray')

        # Plot points in different colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=10, label=f'Point {i+1}')  # Corrected to plot y, x order
        # plt.legend()
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    plt.savefig('pre_test.png')



    binary_mask = np.expand_dims(binary_mask, axis=-1)  # Now shape is (1024, 1024, 1)
    binary_mask = binary_mask.transpose((2, 0, 1))
    points = np.expand_dims(points, axis=1)

    # Return the image, binarized mask, points, and number of masks
    return img, binary_mask, points, len(inds)

# Visualize the data
#Img1, masks1, points1, num_masks = read_batch(train_data, visualize_data=True)


# Set up sam2

def adjust_model_for_grayscale(model):
    # Identify the first convolutional layer dynamically
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv_layer_name = name
            first_conv_layer = module
            break
    
    # Modify the first convolutional layer to accept single-channel input
    new_conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=first_conv_layer.bias is not None
    )
    
    with torch.no_grad():
        # Initialize the weights of the new conv layer
        new_conv_layer.weight[:] = first_conv_layer.weight.mean(dim=1, keepdim=True)
        if first_conv_layer.bias is not None:
            new_conv_layer.bias[:] = first_conv_layer.bias

    # Replace the first convolutional layer with the new layer
    parent_module = model
    submodule_names = first_conv_layer_name.split('.')
    for submodule_name in submodule_names[:-1]:
        parent_module = getattr(parent_module, submodule_name)
    
    setattr(parent_module, submodule_names[-1], new_conv_layer)
    
    return model



sam2_checkpoint = "sam2_hiera_tiny.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
model_cfg = "sam2_hiera_t.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]

# Adjust the SAM2 model for grayscale input
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')

#sam2_model = adjust_model_for_grayscale(sam2_model)

predictor = SAM2ImagePredictor(sam2_model)





# Train mask decoder.
predictor.model.sam_mask_decoder.train(True)

# Train prompt encoder.
predictor.model.sam_prompt_encoder.train(True)

# Configure optimizer.
optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=0.001,weight_decay=1e-4) #1e-5, weight_decay = 4e-5

# # Mix precision.
scaler = torch.cuda.amp.GradScaler()


# No. of steps to train the model.
NO_OF_STEPS = 2000 # @param 

# Fine-tuned model name.
FINE_TUNED_MODEL_NAME = "fine_tuned_sam2"


# Initialize scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2) # 500 , 250, gamma = 0.1
accumulation_steps = 4  # Number of steps to accumulate gradients before updating

# Initialize mean_iou
mean_iou = 0

for step in range(1, NO_OF_STEPS + 1):
    print(step)
    with torch.cuda.amp.autocast():
        image, mask, input_point, num_masks = read_batch(train_data, visualize_data=False)
        
        if image is None or mask is None or num_masks == 0:
            continue

        input_label = np.ones((num_masks, 1))
        if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
            continue

        if input_point.size == 0 or input_label.size == 0:
            continue

        # image = image.copy()
        predictor.set_image(image)

        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
        if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
            continue

        # print(unnorm_coords.shape)
        # print(labels.shape)
        
        # sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        #    points=(unnorm_coords, labels), boxes=None, masks=None,
        # )

        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
           points=None, boxes=None, masks=None,
        )

        torch.cuda.empty_cache()  # Clear cache before operation
    
        batched_mode = unnorm_coords.shape[0] > 1
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
        
        
        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
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
