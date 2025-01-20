import numpy as np
import os
import tifffile 
from IPython import embed
import cv2
import matplotlib.pyplot as plt


# read three channel tif images and convert to np array
def tiff_to_array(tiff_path):
    with tifffile.TiffFile(tiff_path) as tiff:
        image = tiff.asarray() 
    return image

# Convert to 8 bit if needed, could expand to include clipping if needed
def convert_to_8bit(image_16):
    max_value = image_16.max() 
    min_value = image_16.min() 
    image_8 = ((image_16-min_value)/(max_value-min_value)*255).astype('uint8')
    return image_8

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


# Get GT images for train and validation dataset
cwd = os.getcwd()
folder_path = cwd + "/patch_256_8bit/" 

# Loop through masks and correponding images
data_dict = {}
index = 0
for dataset in os.listdir(folder_path):
    dataset_dir = os.path.join(folder_path,dataset)
    for item in os.listdir(dataset_dir):
        if item == 'masks':
            mask_dir = os.path.join(dataset_dir,'masks')
            for mask_img in os.listdir(mask_dir):
                mask_patch = cv2.imread(os.path.join(mask_dir,mask_img))
                contours, _ = cv2.findContours(mask_patch[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour,True)     # Returns 0 if mask is cut
                    mask_len = len(contour)
                    cut_status = check_if_cut(contour, mask_patch.shape[0])
                    # cut_status = perimeter==0.0
                    circularity = get_circularity(perimeter, area)
                    data_dict[index] = {'dataset':dataset, 
                                        'area' : area,
                                        'perimeter' : perimeter, 
                                        'mask_length': mask_len,
                                         'is_cut': cut_status,
                                         'circularity': circularity
                                         }
                    index += 1

                

sets = ['train', 'val', 'test']
for set in sets:
    areas = [entry['area'] for entry in data_dict.values() if entry['dataset'] == set]
    plt.subplot(1, 3, 1)
    plt.boxplot(np.array(areas))
    count = len(areas)
    print(f'Stats of {count} masks in {set} image patches:')
    print(f'Average area: {sum(areas)/count}')
    perimeters = [entry['perimeter'] for entry in data_dict.values() if entry['dataset'] == set]
    plt.subplot(1, 3, 2)
    plt.boxplot(np.array(perimeters))
    plt.title('Perimeter')
    print(f'Average perimeter: {sum(perimeters)/count}')
    cut_mask = [entry['is_cut'] for entry in data_dict.values() if entry['dataset'] == set]
    print(f'Number of cut masks: {sum(cut_mask)}')
    circularity = [entry['circularity'] for entry in data_dict.values() if entry['dataset'] == set]
    print(f'Average circularity: {sum(circularity)/(count-sum(cut_mask))}')
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.boxplot(np.array(areas))
    ax1.set_title('Area')

    ax2.boxplot(np.array(circularity))
    ax2.set_title('Circularity')

    ax3.boxplot(np.array(perimeters))
    ax3.set_title('Perimeter')
    
    plt.savefig(f'stats_{set}.png')
    plt.cla()

    plt.figure()
    plt.scatter(circularity, areas)
    plt.title('Area vs. circularity')
    plt.savefig(f'circ_vs_area_{set}.png')
    plt.cla()




