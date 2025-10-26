import os 
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

def iou(mask1, mask2):
    TP = np.sum((mask1 == 255) & (mask2 == 255))
    FP = np.sum((mask1 == 0) & (mask2 == 255))
    TN = np.sum((mask1 == 0) & (mask2 == 0))
    FN = np.sum((mask1 == 255) & (mask2 == 0))
    iou = TP / (TP + FP + FN)
    return iou

# data folder 
data_folder = "../../../data/patch_256_8bit/test"
bbox_folder = "../results/train_patches/bbox"
file_count = sum(1 for item in os.listdir(bbox_folder) if os.path.isfile(os.path.join(bbox_folder, item)))

# initiate sam model
sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")
predictor = SamPredictor(sam)
predictor.set_image(img_3d)


for i, img_id in enumerate(os.listdir(data_folder)):
    img = cv2.imread(os.path.join(data_folder,'images',img_id))
    feature_dict = 'read csv'
    gt_mask = cv2.imread(os.path.join(data_folder,'masks',img_id[:-4]+'_masks.png'))
    for feature_nb in feature_dict:
        bbox = feature_nb['bbox']

        new_mask, scores, logits = predictor.predict(
            # point_coords=input_point,
            # point_labels=input_label,
            mask_input=bbox,
            multimask_output=True,
        )
        
        # Plot contours of mask
        score_1 = np.uint8(new_mask[0,:,:]*1)
        gt_bbox = gt_mask[bbox]
        iou = iou(img[bbox], gt_bbox)
        if iou>lim:
            mask = mask+new_mask # set max to 1 somehow



        score_2 = np.uint8(new_mask[1,:,:]*1)
        contours, _ = cv2.findContours(score_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        contour = contours[0].squeeze()

        score_3 = np.uint8(new_mask[2,:,:]*1)
        contours, _ = cv2.findContours(score_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        contour = contours[0].squeeze()
        
        if contour.ndim >= 2 and min_area_sam < cv2.contourArea(contours[0]) < max_area_sam:
            if cv2.pointPolygonTest(contour, local_points[0], measureDist=False) > 0:
                contour = contours[0].squeeze()
                
            #if contour.ndim >= 2 and contour.shape[1] < 25:
                contour = np.vstack([contour, contour[0]]) # close contours
                axes[3].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=1)
                axes[3].set_title('Third best score')
                mask = cv2.fillPoly(mask, [contour], 255)

        # = mask + new_mask