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



# Get GT images for train and validation dataset
cwd = os.getcwd()
folder_dir = cwd + "/full_size/" 

# Loop through masks and correponding images
data_dict = {}
index = 0
for data_set in os.listdir(folder_dir):
    dataset_dir = os.path.join(folder_dir,data_set)
    for file_name in os.listdir(dataset_dir):
        if file_name[-4:] == '.tif':
            continue
        elif file_name[-4:] == '.npy':
            seg_path = os.path.join(dataset_dir, file_name)
            seg_object = np.load(seg_path, allow_pickle=True).item()
            fed_state = file_name[33] # FIXME: the position of state may change with image name
   
            # make binary
            binary_mask = seg_object['masks']
            binary_mask[binary_mask > 0] = 255
            binary_mask = binary_mask.astype(np.uint8)

            # Loop with sliding window to make patches
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour,True)     # Returns 0 if mask is cut
                mask_len = len(contour)
                cut_status = check_if_cut(contour, binary_mask.shape[0])

                circularity = get_circularity(perimeter, area)
                data_dict[index] = {'dataset':data_set, 
                                    'area' : area,
                                    'perimeter' : perimeter, 
                                    'mask_length': mask_len,
                                    'is_cut': cut_status,
                                    'circularity': circularity,
                                    'fed_state': fed_state,
                                    }
                index += 1
   
# Get info for each fed state
# filtered_dict = {key: entry for key, entry in data_dict.items() if entry['circularity'] >= 0.5}
filtered_dict = {key: entry for key, entry in data_dict.items()}
data_dict = filtered_dict
sets = ['S', 'R', 'F']
for set in sets:

    # areas = [entry['area'] for entry in data_dict.values() if entry['fed_state'] == set]
    areas = [entry['area'] for entry in data_dict.values() if entry.get('fed_state', '').upper() == set]

    count = len(areas)
    print(f'\nStats of {count} masks in {set} image patches:')

    print(f'Average area: {sum(areas)/count}')
    # perimeters = [entry['perimeter'] for entry in data_dict.values() if entry['fed_state'] == set]
    perimeters = [entry['perimeter'] for entry in data_dict.values() if entry.get('fed_state', '').upper() == set]
    print(f'Average perimeter: {sum(perimeters)/count}')
    # cut_mask = [entry['is_cut'] for entry in data_dict.values() if entry['fed_state'] == set]
    cut_mask = [entry['is_cut'] for entry in data_dict.values() if entry.get('fed_state', '').upper() == set]
    print(f'Number of cut masks: {sum(cut_mask)}')
    circularity = [entry['circularity'] for entry in data_dict.values() if entry.get('fed_state', '').upper() == set]
    
    # circularity = [entry['circularity'] for entry in data_dict.values() if entry['fed_state'] == set]
    print(f'Average circularity: {sum(circularity)/(count-sum(cut_mask))}')


    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.violinplot(np.array(areas))
    ax1.set_title('Area')
    ax1.yaxis.grid(True)
    

    ax2.violinplot(np.array(circularity))
    ax2.set_title('Circularity')
    ax2.yaxis.grid(True)

    ax3.violinplot(np.array(perimeters))
    ax3.set_title('Perimeter')
    ax3.yaxis.grid(True)

    
    # Plot circ vs area
    plt.savefig(f'plots/stats_{set}.png')
    plt.cla()

    plt.figure()
    plt.scatter(circularity, areas, s=10)
    plt.title('Area vs. circularity')
    plt.grid()
    plt.ylim(0,3500)
    plt.savefig(f'plots/circ_vs_area_{set}.png')
    plt.cla()

# Filter on metrics instead, to make plots across all fed state
# Filter out the spots with little circularity, since they are probably merged spots. Must mention this in report.
# filtered_dict = {key: entry for key, entry in data_dict.items()}
# filtered_dict = {key: entry for key, entry in data_dict.items() if entry['circularity'] >= 0.2}
removed_entries = {key: entry for key, entry in data_dict.items() if entry['area'] > 700}
print(f'Removed {len(removed_entries)} entries')
filtered_dict = {key: entry for key, entry in data_dict.items() if entry['area'] <= 700}
data_dict = filtered_dict
sets = ['area', 'perimeter', 'circularity']
for set in sets:

    values_f = [entry[set] for entry in data_dict.values() if entry['fed_state'] == 'F']
    values_s = [entry[set] for entry in data_dict.values() if entry['fed_state'] == 'S']
    values_r = [entry[set] for entry in data_dict.values() if entry['fed_state'] == 'R']


    values_list= [values_f,values_r,values_s]
    ymax = max(max(sublist) for sublist in values_list)
    ymax = ymax + ymax/20
    ymin = min(min(sublist) for sublist in values_list)
  
   # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.violinplot(np.array(values_f))
    ax1.set_title('Fed')
    ax1.yaxis.grid(True)
    ax1.set_ylabel(set)
    ax1.set_ylim(ymin,ymax)

    ax2.violinplot(np.array(values_s))
    ax2.set_title('Starved')
    ax2.yaxis.grid(True)
    ax2.set_ylim(ymin,ymax)

    ax3.violinplot(np.array(values_r))
    ax3.set_title('Refed')
    ax3.yaxis.grid(True)
    ax3.set_ylim(ymin,ymax)

    
    # Plot circ vs area
    plt.savefig(f'plots/stats_{set}.png')
    plt.cla()