import sys
import pandas as pd
import cv2
from IPython import embed
from scipy.ndimage import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os



# Initiate true positives, false positives and false negatives counters
total_tp = 0
total_fp = 0
total_fn = 0

# Folders
results_folder = '../results/fullsize'
mask_folder = "../../../data/fullsize/test/masks"
img_folder = "../../../data/fullsize/test/images"

# Set min peak distance from base level to include in results
min_z_dist = [100, 100, 100]


# Loop through test images
for i, file in enumerate(os.listdir(results_folder)):
    result_file = os.path.join(results_folder, file)
    print(f'Processing image {file[:-4]}')

    # Load results 
    df = pd.read_csv(result_file, delimiter=',', header=None)
    column_names = ['x','y','base_level', 'z_dist']
    #column_names = ['y','x','base_level', 'z_dist']
    df.columns = column_names
    # Load gt masks
    mask_path = os.path.join(mask_folder, f"{file[:-4]}_masks.png")
    masks_uint8 = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    masks = masks_uint8/255
    # Load img
    img_path = os.path.join(img_folder, f"{file[:-4]}.png")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


    # filter out peaks that have too low dist from base
    filtered_df = df[df['z_dist'] >= int(min_z_dist[i])]

    df = filtered_df.copy()

    # Get false positive by lopping through all peaks and check the corresponding mask value
    fp = len(df) # initiate false positives as all points
    fp_list = [] # To store coordinates of false positive for plotting
    df['correct'] = 0 # Count for performance metrics

    # Loop over all peaks
    for index, row in df.iterrows():
        #gt_value = masks[int(row['x']), int(row['y'])]
        gt_value = masks[int(row['y']), int(row['x'])]
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
        coords_df = pd.DataFrame(coords, columns=['y', 'x']) # convert to df for cooordinate comparison
        #coords_df = pd.DataFrame(coords, columns=['x', 'y']) # convert to df for cooordinate comparison
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

    plt.figure(figsize=(20,20))
    plt.imshow(masks, cmap=cmap, interpolation='nearest')
    # plt.imshow(ann_map)
    # for index, row in df.iterrows():
    #     plt.plot(row['x'], row['y'], marker='.', markersize=6, c='r')
    for row in fp_list:
        plt.plot(row[0], row[1], marker='.', markersize=4, c='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'../plots/fullsize/{file[:-4]}_eval.png')
    plt.clf()


    plt.figure(figsize=(20,20))
    plt.imshow(img, cmap='gray')
    for index, row in df.iterrows():
        plt.plot(row['x'], row['y'], marker='.', markersize=4, c='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'../plots/fullsize/{file[:-4]}_predictions.png')
    plt.clf()

    print(f'precision proxy: {precision*100}\nrecall: {recall*100}')
    print(f'tp: {int(tp)}/{value-small_gt_masks}')
    print(f'fn: {int(fn)}/{value-small_gt_masks}')
    print(f'fp: {int(fp)}')
    print(f'F1 score: {100*2*tp/(2*tp+fp + fn)}') 

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
    plt.savefig(f'../plots/fullsize/{file[:-4]}_base_vs_dist.png')
    # print(f'{multi_peaks} out of {len(contours)} gt masks had more than one peak')
    # print(f'masks found with cv2: {len(contours)}\nmasks found with scimage: {num_features} ')
    total_fp = total_fp + fp
    total_fn = total_fn + fn
    total_tp = total_tp + tp

total_precision = total_tp / (total_tp + total_fp)
total_recall = total_tp / (total_tp+total_fn) 
total_f1 = 2*total_tp/(2*total_tp+total_fp + total_fn)

print(f'Total all 3 images')
print(f'Total precision: {total_precision*100}')
print(f'Total recall: {total_recall*100}')
print(f'Total F1 score: {total_f1*100}')
