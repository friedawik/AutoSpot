import sys
import pandas as pd
import cv2
from IPython import embed
from scipy.ndimage import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from spot_detection_tool import SpotDetector

""" At the moment not working to analyse fullsized images, or just very slow. 
Therefore, they must first been divided into patches and then patched together again.
Use performance_all for this. """

# get training image
image_name = f'MF_MaxIP_3ch_2_000_230623_544_84_F_XY4'
image_path = '../../../data/fullsize/test/images/' + image_name + '.png'
mask_path = '../../../data/fullsize/test/masks/' + image_name+ '_masks.png'

# Open images and gt masks
img = cv2.imread(image_path,  cv2.IMREAD_UNCHANGED)

masks = cv2.imread(mask_path)

min_area= 3
max_area= 200
levels = 5
min_z = 5
spots=SpotDetector(img,contour_levels=levels, min_area=min_area, max_area=max_area,min_circularity = 0.3, z_min_dist = min_z)
df = spots.plot_centroids()

#filtered_df = df[df['z_dist'] >= int(min_z_dist)]
    #filtered_df = filtered_df[filtered_df['prominence'] >= int(min_prominence)]

#df = filtered_df.copy()


# Get false positive by lopping through all peaks and check the corresponding mask value
fp = len(df) # initiate false positives as all points
fp_list = [] # To store coordinates of false positive for plotting
df['correct'] = 0 # Count for performance metrics

# Loop over all peaks
for index, row in df.iterrows():
    gt_value = masks[int(row['x']), int(row['y'])]
    #gt_value = masks[int(row['y']), int(row['x'])]
    #tp += gt_value # add if inside mask
    fp = fp - gt_value # subtract value 1 if inside mask 
    if gt_value == 1:
        df.at[index, 'correct'] = 1
    if gt_value == 0:
        fp_list.append([int(row['x']), int(row['y'])])
        df.at[index, 'correct'] = 0


    # Get mask count
    ann_map, num_features = label(masks)

    tp = 0  # True positives are masks that have one or more peaks inside
    fn = 0  # False negatives are masks that have no peak in them

    small_gt_masks = 0
    multi_peaks = 0 # Count masks that have more than 1 peaks in them
    for value in range(1,num_features+1):
        coords = np.argwhere(ann_map == value) # get all cordinates of a mask
        #coords_df = pd.DataFrame(coords, columns=['y', 'x']) # convert to df for cooordinate comparison
        coords_df = pd.DataFrame(coords, columns=['x', 'y']) # convert to df for cooordinate comparison
        # Merge to find matching coordinates
        merged_df = pd.merge(df, coords_df, on=['x', 'y'], how='inner')
        # merged_df = pd.merge(df, coords_df, on=['y', 'x'], how='inner')

        # Empty df means no equal coordinates
        if len(merged_df)<1:
            fn = fn + 1
            masks[coords[:, 0], coords[:, 1]] = 2

        # One row means exacly one peak in mask
        elif len(merged_df)==1:
            tp = tp + 1
            multi_peaks = multi_peaks + 1
            masks[coords[:, 0], coords[:, 1]] = 1
        
        # More than one row means multiple peaks in mask
        else:
            tp = tp + 1
            masks[coords[:, 0], coords[:, 1]] = 1


    precision = tp / (tp + fp)
    recall = tp / (tp+fn)  


    # Create a custom colormap
    colors = ['black', 'yellow', 'red']
    cmap = ListedColormap(colors)

    plt.figure()
    plt.imshow(masks, cmap=cmap, interpolation='nearest')
    # plt.imshow(ann_map)
    # for index, row in df.iterrows():
    #     plt.plot(row['x'], row['y'], marker='.', markersize=6, c='r')
    for row in fp_list:
        plt.plot(row[0], row[1], marker='.', markersize=1, c='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'../plots/{image_name}_eval.png')

    print(f'precision proxy: {precision*100}\nrecall: {recall*100}')
    print(f'tp: {int(tp)}/{value-small_gt_masks}')
    print(f'fn: {int(fn)}/{value-small_gt_masks}')
    print(f'fp: {int(fp)}')

    plt.cla()
    plt.figure()
    true_df = df[df['correct'] == 1]
    false_df = df[df['correct'] == 0]
    for index, row in true_df.iterrows():
        plt.plot(row['z_dist'], row['base_level'], marker='.', markersize=4, c='r')
    for index, row in false_df.iterrows():
        plt.plot(row['z_dist'], row['base_level'], marker='.', markersize=4, c='b')

    plt.xlabel('z_dist')
    plt.ylabel('base_level')
    plt.grid()
    plt.savefig('test.png')
    # print(f'{multi_peaks} out of {len(contours)} gt masks had more than one peak')
    # print(f'masks found with cv2: {len(contours)}\nmasks found with scimage: {num_features} ')
    total_fp = total_fp + fp
    total_fn = total_fn + fn
    total_tp = total_tp + tp

total_precision = total_tp / (total_tp + total_fp)
total_recall = total_tp / (total_tp+total_fn) 

print(f'Total all 3 images')
print(f'Total precision: {total_precision}')
print(f'Total recall: {total_recall}')
