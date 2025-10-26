import sys
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label

"""
Code to calculate precision and recall of single 256*256 patches, where TP is 
any peak inside of a gt mask and FP is any mask outside of gt mask. FP is any 
mask that does not contain a detected peak.

The results are plotted as: 
1) GT masks corresponding to TP or FP colored in different colors.
2) Plot of prominence vs. elevation.

Args:
    min_elevation (int): The minimum elevation threshold used for visualization, specified as a command-line argument.
    min_prominence (int): The minimum prominence value for features to be included in the visualization, also specified as a command-line argument.
"""

def load_data(img_id, min_elevation, min_prominence=None):
    """Load and filter data."""
    df = pd.read_csv(f'../results/results_patch/{img_id}.txt', delimiter=',')
    df[['x', 'y']] = df[['x', 'y']].astype(int)
    masks_uint8 = cv2.imread(f'../../data/patch_256/test/masks/{img_id}_masks.png', cv2.IMREAD_UNCHANGED)
    masks = masks_uint8 / 255

    df = df[df['elevation'] >= int(min_elevation)]
    if min_prominence is not None:
        df = df[df['prominence'] >= int(min_prominence)]

    return df, masks

def evaluate_results(df, masks):
    """Evaluate results and calculate metrics."""

    # calculate fp, start with all points as fp and remove
    fp = len(df)
    fp_list = []
    df['correct'] = 0

    # go through df and check if peaks are 0 or 1 in mask 
    for index, row in df.iterrows():
        gt_value = masks[int(row['y']), int(row['x'])]
        fp -= gt_value
        df.at[index, 'correct'] = int(gt_value)
        if gt_value == 0:
            fp_list.append([int(row['x']), int(row['y'])])

    #  get tp and fn while checking if there are multiple peaks in onw gt mask
    ann_map, num_features = label(masks)
    tp, fn, multi_peaks = 0, 0, 0

    for value in range(1, num_features + 1):
        # get coordinates that are both part of each mask and have a peak
        coords = np.argwhere(ann_map == value)
        coords_df = pd.DataFrame(coords, columns=['y', 'x'])
        merged_df = pd.merge(df, coords_df, on=['x', 'y'], how='inner')
        # if no such pixels, it is a fp
        if len(merged_df) < 1:
            fn += 1
            masks[coords[:, 0], coords[:, 1]] = 2
        # if peaks and mask pixels are same, count as tp
        else:
            tp += 1
            masks[coords[:, 0], coords[:, 1]] = 1
            # if multiple peaks in same mask, register in multi_peaks
            if len(merged_df) > 1:
                multi_peaks += 1

    if (tp+fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if (tp+fn) != 0:
        recall = tp / (tp + fn)
    else:
        recall=0

    return precision, recall, tp, fn, fp, fp_list, masks, multi_peaks, num_features

def plot_results(masks, fp_list, img_id):
    """Plot and save evaluation results."""
    colors = ['black', 'yellow', 'red']
    cmap = ListedColormap(colors)

    plt.figure()
    plt.imshow(masks, cmap=cmap, interpolation='nearest')
    for row in fp_list:
        plt.plot(row[0], row[1], marker='.', markersize=6, c='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'../plots/{img_id}_eval.png')
    plt.close()

def plot_prominence_vs_elevation(df, img_id):
    """Plot prominence vs elevation."""
    plt.figure()
    true_df = df[df['correct'] == 1]
    false_df = df[df['correct'] == 0]
    plt.plot(true_df['elevation'], true_df['prominence'], 'r.', markersize=4, alpha=0.5)
    plt.plot(false_df['elevation'], false_df['prominence'], 'b.', markersize=4, alpha=0.5)
    plt.xlabel('elevation')
    plt.ylabel('prominence')
    plt.grid()
    plt.savefig(f'../plots/{img_id}_prom_vs_el.png')
    plt.close()

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <img_id> <min_elevation> [min_prominence]")
        sys.exit(1)

    img_id = sys.argv[1]
    min_elevation = sys.argv[2]
    min_prominence = sys.argv[3] if len(sys.argv) > 3 else None

    df, masks = load_data(img_id, min_elevation, min_prominence)
    precision, recall, tp, fn, fp, fp_list, masks, multi_peaks, num_features = evaluate_results(df, masks)

    plot_results(masks, fp_list, img_id)
    plot_prominence_vs_elevation(df, img_id)

    print(f'Precision proxy: {precision*100:.2f}%')
    print(f'Recall proxy: {recall*100:.2f}%')
    print(f'TP: {int(tp)}/{num_features}')
    print(f'FN: {int(fn)}/{num_features}')
    print(f'FP: {int(fp)}')
    print(f'Multi-peaks: {multi_peaks} out of {num_features} GT masks')

if __name__ == "__main__":
    main()