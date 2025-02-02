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

def read_input(path):
    bbox_list = []
    pos_list = []
    neg_list = []
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row if present
        for row in csv_reader:
            bbox = [int(value) for value in row]  # Convert string values to integers
            bbox_list.append(bbox[:4])
            pos_list.append(bbox[-4:-2])
            neg_list.append(bbox[-2:])
    
    bbox_arr = np.array(bbox_list)
    pos_arr = np.array(pos_list)
    neg_arr = np.array(neg_list)

    return bbox_arr, pos_arr, neg_arr

def read_mask(gt_mask):
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    individual_masks = []
    for contour in contours:
        mask = np.zeros(gt_mask.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        mask = np.expand_dims(mask, axis=0)
        individual_masks.append(mask)
    return individual_masks

def plot_mask(mask):
    plt.imshow(mask[0,:,:])
    plt.savefig('test.png')
    plt.clf()

def plot_input(mask, pos_points=None, neg_points=None, bbox=None):
    plt.imshow(mask)
    plt.scatter(pos_points[:,:1],pos_points[:,1:], c='b', s=4)
    plt.scatter(neg_points[:,:1],neg_points[:,1:], c='r', s=4)
    plt.savefig('test.png')
    plt.clf()

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
bbox_folder = os.path.expanduser("~/master/git/spot_segmentation/contours/results/points_patches")

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


for i, img_id in enumerate(os.listdir(data_folder+'/images')):
    # img_id = 'MF_MaxIP_3ch_2_000_230623_544_84_R1h_XY5_x3_y3.png'
    if (i+1)%10==0:
        print(f'Processing image {img_id}: {i+1}/{file_count}')
    bbox_path = os.path.join(results_folder,'points_patches',img_id[:-3] + 'csv')
    bbox_arr, pos_arr, neg_arr = read_input(bbox_path)
    if len(bbox_arr)==0:
        print("There are no bbox in this image.")
        continue
    pos_labels = np.ones(len(pos_arr))
    #pos_labels = np.ones((len(pos_arr),1))
    neg_labels = np.zeros(len(neg_arr))
    labels = np.concatenate((pos_labels, neg_labels), axis=0)
    points = np.concatenate((pos_arr, neg_arr), axis=0)

    img = cv2.imread(os.path.join(data_folder,'images',img_id))
    #predictor.set_image(img)
    input_patch = cv2.imread(os.path.join(results_folder, 'patches',img_id))
    gt_mask = cv2.imread(os.path.join(data_folder,'masks',img_id[:-4]+'_masks.png'))
    #new_mask = np.expand_dims(gt_mask[:,:,0]/255, axis=0)
    plot_input(input_patch[:,:,0], pos_arr, neg_arr, bbox=None)

    masks_individual = read_mask(input_patch[:,:,0])
    mask_test = input_patch[:64, 50:114][:,:,0]
    img_test = img[:64, 50:114]
    
    mask_test = np.expand_dims(mask_test, axis=0)
    img_test = np.expand_dims(img_test, axis=0)
    reshaped_img = np.transpose(img_test, (2, 0, 1))
    embed()
    predictor.set_image(img_test)
    print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)
    embed()


    masks, scores, _ = predictor.predict(
    point_coords=np.array([[6,  77]]),
    point_labels=np.array([1]),
    box=None,
    mask_input= None,
    multimask_output=False,
    )
    plot_mask(masks[0,:,:])
    
    filtered_masks = masks[scores > 0.5]
    pred_masks = merge_binary_images(filtered_masks)



        # max_len = 5
        # count_short_arr = int(len(bbox_arr)/max_len)+1
        # pred_masks = np.zeros((img.shape[0], img.shape[1]))
        # torch.cuda.empty_cache()
        # for i in range (count_short_arr):
        #     embed()
            
        #     masks, scores, _ = predictor.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=bbox_arr[i*max_len:(i+1)*max_len],
        #     multimask_output=False,
        #     )
            
        #     filtered_masks = masks[scores > 0.5]
        #     pred_masks_short_arr = merge_binary_images(filtered_masks)
        #     pred_masks = cv2.bitwise_or(pred_masks_short_arr, pred_masks)
            
            # print(f'Prediction done in {count_short_arr} steps.')
        #print(f'skipping img {img_id}')
        #continue
    # Only use masks with high enough score
    #filtered_masks = masks[scores[:,2] > 0.5]

    filtered_masks = masks[scores > 0.5]
    pred_masks = merge_binary_images(filtered_masks)
    #pred_masks = merge_binary_images(filtered_masks[:,2,:,:])
    
    pred_masks_swapped = np.swapaxes(pred_masks, 0, 1)
    plot_prediction(img, pred_masks_swapped, gt_mask, bbox_arr, input_patch, img_id[:-4])
    
    save_path = os.path.join(results_folder, 'sam2_patches_points', img_id )
    cv2.imwrite(save_path, pred_masks_swapped*255)



