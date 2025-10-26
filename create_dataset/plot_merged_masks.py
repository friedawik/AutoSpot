import numpy as np
import os
import tifffile 
from IPython import embed
from cellpose import models, plot, utils, io
from cellpose.io import imread
import cv2
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

def check_multiple_maxima(image, min_distance=20, threshold_rel=0.8):
    coordinates = peak_local_max(image, min_distance=min_distance, threshold_rel=threshold_rel)
    return len(coordinates) 

# read three channel tif images and convert to np array
def tiff_to_array(tiff_path):
    with tifffile.TiffFile(tiff_path) as tiff:
        image = tiff.asarray() 
    return image

def check_if_cut(cnt, patch_shape):
    cut_mask = False
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    if leftmost[0] == 0 or topmost[0] == 0 or rightmost[1] == patch_shape or bottommost[1] == patch_shape:
        cut_mask = True
    return cut_mask

def get_circularity(perimeter, area):
    if perimeter>0.0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else:
        circularity = 0
    return circularity

# Convert to 8 bit if needed, could expand to include clipping if needed
def convert_to_8bit(image_16):
    max_value = image_16.max() 
    min_value = image_16.min() 
    image_8 = ((image_16-min_value)/(max_value-min_value)*255).astype('uint8')
    return image_8



# Get test images paths 
# folder_dir = os.path.join(os.getcwd(), 'full_size','test')
folder_dir = os.path.join(os.getcwd(), 'full_size','train_val')
f_img_dir = os.path.join(folder_dir,'MF_MaxIP_3ch_2_000_230623_544_84_F_XY4_seg.npy')
s4h_img_dir = os.path.join(folder_dir,'MF_MaxIP_3ch_2_000_230623_544_84_R1h_XY5_seg.npy')
r1h_mask_dir = os.path.join(folder_dir,'MF_MaxIP_3ch_2_000_230623_544_84_S4h_XY2_seg.npy')
r1h_img_dir = os.path.join(folder_dir,'MF_MaxIP_3ch_2_000_230623_544_84_S4h_XY2.tif')

r1h_mask_dir = os.path.join(folder_dir,'MF_MaxIP_3ch_2_000_230623_544_84_R1h_XY1_seg.npy')
r1h_img_dir = os.path.join(folder_dir,'MF_MaxIP_3ch_2_000_230623_544_84_R1h_XY1.tif')



seg_object = np.load(r1h_mask_dir, allow_pickle=True).item()
img = tiff_to_array(r1h_img_dir)
img = img[:,:,1]
# img_8bit = convert_to_8bit(img[:,:,1])

# make binary
binary_mask = seg_object['masks']
binary_mask[binary_mask > 0] = 255
binary_mask = binary_mask.astype(np.uint8)

# Loop with sliding window to make patches
bbox_list = []
contour_list = []
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area>900:
        rect = cv2.boundingRect(contour)
        bbox_list.append(rect)
        contour_list.append(contour)
        


# img_bbox = img
# mask_bbox = binary_mask
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig, axs = plt.subplots(3, 2, figsize=(10, 15))

plt.tight_layout()
dim = 80
buf = 5
for i in range(3):
    x,y,w,h = bbox_list[i]
    img_bbox = img[y-buf:y+dim, x-buf:x+dim]
    mask_bbox = binary_mask[y-buf:y+dim, x-buf:x+dim]
    axs[i, 0].imshow(img_bbox, cmap='gray')
    axs[i, 0].set_title('The image')
    axs[i, 1].imshow(mask_bbox, cmap='gray')
    axs[i, 1].set_title('The GT mask')

plt.savefig('plots/merged_spots.png')
plt.cla()

# Plot contour lines instead

dim = 80
buf = 5
image_cont = img.copy()
image_cont = convert_to_8bit(image_cont)
# cv2.drawContours(image=image_cont, contours=contour_list,contourIdx=-1, color=(255, 0, 0), thickness=1)
cv2.drawContours(image=image_cont, contours=contours,contourIdx=-1, color=(255, 255, 0), thickness=1)
# cv2.drawContours(image=image_cont, contours=contour_list,contourIdx=-1, color=(0, 0, 255), thickness=1)


fig, axs = plt.subplots(2, 3, figsize=(15, 10))
plt.tight_layout()
ind = 2
for i in range (2):
    for j in range(3):
        x,y,w,h = bbox_list[ind]
        img_bbox = image_cont[y-buf:y+dim, x-buf:x+dim]
        axs[i, j].imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))
        ind+=1


plt.savefig('plots/merged_spots_cont.png')



