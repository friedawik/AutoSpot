import sys
import pandas as pd
import cv2
from IPython import embed
from scipy.ndimage import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os



min_elevation = sys.argv[1]
min_prominence = sys.argv[2]


total_tp = 0
total_fp = 0
total_fn = 0

results_dir = 'results_fullsize'
for file in os.listdir(results_dir):
    result_file = os.path.join(results_dir, file)
    print(f'Image {file[:-4]}')
    # min_elevation = input(f"Give min elevation: ")
    # min_prominence = input(f"Give min prominence: ")
    # Load results and GT mask
    df = pd.read_csv(result_file, delimiter=',')
    df[['x', 'y']] = df[['x', 'y']].astype(int)
    masks_uint8 = cv2.imread(f'fullsize_images/masks/{file[:-4]}_masks.png', cv2.IMREAD_UNCHANGED)
    masks = masks_uint8/255

    # filter low elevation
    filtered_df = df[df['elevation'] >= int(min_elevation)]
    filtered_df = filtered_df[filtered_df['prominence'] >= int(min_prominence)]

    df = filtered_df.copy()

    # Get false positive by lopping through all peaks found by prominence and check the corresponding mask value

    fp = len(df) # initiate false positives as all points
    fp_list = []
    df['correct'] = 0

    for index, row in df.iterrows():
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

    tp = 0  # True positives are masks that have one or more peaks found by prominence
    fn = 0  # False negatives are masks that have no peak in them

    small_gt_masks = 0
    multi_peaks = 0 # Count masks that have more than 1 peaks in them
    for value in range(1,num_features+1):
        coords = np.argwhere(ann_map == value)
        # if len(coords)<8:
        #     small_gt_masks = small_gt_masks + 1
        #     embed()
        #     # plt.plot(coords[0][0], coords[0][1], markersize=6)
        #     continue

        # Convert the array to a DataFrame for easier comparison
        # coords_df = pd.DataFrame(coords, columns=['x', 'y'])
        coords_df = pd.DataFrame(coords, columns=['y', 'x'])
        # Merge to find matching coordinates
        merged_df = pd.merge(df, coords_df, on=['x', 'y'], how='inner')
        # merged_df = pd.merge(df, coords_df, on=['y', 'x'], how='inner')

        if len(merged_df)<1:
            fn = fn + 1
            masks[coords[:, 0], coords[:, 1]] = 2
        elif len(merged_df)==1:
            tp = tp + 1
            multi_peaks = multi_peaks + 1
            masks[coords[:, 0], coords[:, 1]] = 1
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
    plt.savefig(f'plots/{file[:-4]}_eval.png')

    print(f'precision proxy: {precision*100}\nrecall: {recall*100}')
    print(f'tp: {int(tp)}/{value-small_gt_masks}')
    print(f'fn: {int(fn)}/{value-small_gt_masks}')
    print(f'fp: {int(fp)}')

    plt.cla()
    plt.figure()
    true_df = df[df['correct'] == 1]
    false_df = df[df['correct'] == 0]
    for index, row in true_df.iterrows():
        plt.plot(row['elevation'], row['prominence'], marker='.', markersize=4, c='r')
    for index, row in false_df.iterrows():
        plt.plot(row['elevation'], row['prominence'], marker='.', markersize=4, c='b')

    plt.xlabel('elevation')
    plt.ylabel('prominence')
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
