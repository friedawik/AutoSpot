import os 
import cv2
import numpy as np
import torch
#from segment_anything import SamPredictor, sam_model_registry
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from IPython import embed
import csv
from tools import plot_prediction
import matplotlib.pyplot as plt

def iou(mask1, mask2):
    TP = np.sum((mask1 == 255) & (mask2 == 255))
    FP = np.sum((mask1 == 0) & (mask2 == 255))
    TN = np.sum((mask1 == 0) & (mask2 == 0))
    FN = np.sum((mask1 == 255) & (mask2 == 0))
    iou = TP / (TP + FP + FN)
    return iou

def read_bbox(path):
    bbox_list = []

    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row if present
        for row in csv_reader:
            bbox = [int(value) for value in row]  # Convert string values to integers
            bbox_list.append(bbox)
    bbox_arr = np.array(bbox_list)
    return bbox_arr

def merge_binary_images(images):
    result = images[0]
    for img in images[1:]:
        result = cv2.bitwise_or(result, img)
    return result

# set device to cuda
if torch.cuda.is_available():
    device = torch.device("cuda")



# data folder 
data_folder = "../../../../master/git/data/patch_256_8bit/test"
#bbox_folder = "../../../../master/git/spot_segmentation/contours/results/bbox_patches"
bbox_folder = os.path.expanduser("~/master/git/spot_segmentation/contours/results/bbox_patches")

results_folder = os.path.expanduser("~/master/git/spot_segmentation/contours/results")
file_count = sum(1 for item in os.listdir(bbox_folder) if os.path.isfile(os.path.join(bbox_folder, item)))

# initiate sam model
sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

# sam2_checkpoint = "sam2.1_hiera_tiny.pt"
# #sam2_checkpoint = "sam2_hiera_tiny.pt"
# model_cfg = "sam2.1_hiera_t.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

#img_id = 'MF_MaxIP_3ch_2_000_230623_544_84_R1h_XY5_x3_y4.png'
img_id = 'MF_MaxIP_3ch_2_000_230623_544_84_S4h_XY2_x2_y4.png'
#for i, img_id in enumerate(os.listdir(data_folder+'/images')):
    # img_id = 'MF_MaxIP_3ch_2_000_230623_544_84_R1h_XY5_x3_y3.png'
# print(f'Processing image {img_id}: {i+1}/{file_count}')
bbox_path = os.path.join(results_folder,'bbox_patches',img_id[:-3] + 'csv')
bbox_arr = read_bbox(bbox_path)
# if len(bbox_arr)==0:
#     print("There are no bbox in this image.")
#     continue

img = cv2.imread(os.path.join(data_folder,'images',img_id))
predictor.set_image(img)
input_patch = cv2.imread(os.path.join(results_folder, 'patches',img_id))
gt_mask = cv2.imread(os.path.join(data_folder,'masks',img_id[:-4]+'_masks.png'))



# try:
masks, scores, _ = predictor.predict(
point_coords=None,
point_labels=None,
box=bbox_arr,
multimask_output=False,
)

filtered_masks = masks[scores > 0.5]
pred_masks = merge_binary_images(filtered_masks)


    # except:
        # max_len = 5
        # count_short_arr = int(len(bbox_arr)/max_len)+1
        # pred_masks = np.zeros((img.shape[0], img.shape[1]))
        # torch.cuda.empty_cache()
        # for j in range (count_short_arr):
        #     embed()
            
        #     masks, scores, _ = predictor.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=bbox_arr[j*max_len:(j+1)*max_len],
        #     multimask_output=False,
        #     )
            
        #     filtered_masks = masks[scores > 0.5]
        #     pred_masks_short_arr = merge_binary_images(filtered_masks)
        #     pred_masks = cv2.bitwise_or(pred_masks_short_arr, pred_masks)
            
            # print(f'Prediction done in {count_short_arr} steps.')
        # print(f'skipping img {img_id}')
        # continue
    # Only use masks with high enough score
    #filtered_masks = masks[scores[:,2] > 0.5]

filtered_masks = masks[scores > 0.5]
pred_masks = merge_binary_images(filtered_masks)
#pred_masks = merge_binary_images(filtered_masks[:,2,:,:])

pred_masks_swapped = np.swapaxes(pred_masks, 0, 1)
plot_prediction(img, pred_masks_swapped, gt_mask, bbox_arr, input_patch, img_id[:-4])
save_path = os.path.join(results_folder, 'sam2_patches', img_id )
cv2.imwrite(save_path, pred_masks_swapped*255)



