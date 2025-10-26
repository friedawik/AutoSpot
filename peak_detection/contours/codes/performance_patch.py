import sys
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label
from IPython import embed

def load_data(img_id, min_z_dist):
    """Load and filter data."""
    column_names = ['x','y','base_level', 'z_dist']
    df = pd.read_csv(f'../results/patches/{img_id}.txt', delimiter=',', header=None)
    df.columns = column_names
    #df[['x', 'y']] = df[['x', 'y']].astype(int)
    masks_uint8 = cv2.imread(f'../../../data/patch_256_8bit/test/masks/{img_id}_masks.png', cv2.IMREAD_UNCHANGED)
    masks = masks_uint8 / 255

    img = cv2.imread(f'../../../data/patch_256_8bit/test/images/{img_id}.png', cv2.IMREAD_UNCHANGED)

    df = df[df['z_dist'] >= int(min_z_dist)]
    return df, masks, img

def evaluate_results(df, masks):
    """Evaluate results and calculate metrics."""
    fp = len(df)
    fp_list = []
    df['correct'] = 0

    for index, row in df.iterrows():
        gt_value = masks[int(row['y']), int(row['x'])]
        fp -= gt_value
        df.at[index, 'correct'] = int(gt_value)
        if gt_value == 0:
            fp_list.append([int(row['x']), int(row['y'])])

    ann_map, num_features = label(masks)
    tp, fn, multi_peaks = 0, 0, 0

    for value in range(1, num_features + 1):
        coords = np.argwhere(ann_map == value)
        coords_df = pd.DataFrame(coords, columns=['y', 'x'])
        merged_df = pd.merge(df, coords_df, on=['x', 'y'], how='inner')

        if len(merged_df) < 1:
            fn += 1
            masks[coords[:, 0], coords[:, 1]] = 2
        else:
            tp += 1
            masks[coords[:, 0], coords[:, 1]] = 1
            if len(merged_df) > 1:
                multi_peaks += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall, tp, fn, fp, fp_list, masks, multi_peaks, num_features

def plot_results(masks, img, fp_list, img_id):
    """Plot and save evaluation results."""
    colors = ['black', 'yellow', 'red']
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    
    for row in fp_list:
        axes[1].plot(row[0], row[1], marker='.', markersize=4, c='b')
    axes[0].imshow(img, cmap='gray')
    axes[1].imshow(masks, cmap=cmap, interpolation='nearest')
    axes[2].imshow(img, cmap='gray')
    #plt.xlabel('x')
    #plt.ylabel('y')
    plt.savefig(f'../plots/patches/{img_id}_eval.png')
    plt.close()

def plot_predictions(img, df, img_id):
    """Plot and save evaluation results."""
    masks_uint8 = cv2.imread(f'../../../data/patch_256_8bit/test/masks/{img_id}_masks.png', cv2.IMREAD_UNCHANGED)
    masks = masks_uint8 / 255

    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=(15,5))
    
    for index, row in df.iterrows():
        plt.plot(row['x'], row['y'], marker='.', markersize=6, c='r')
    axes[0].imshow(img, cmap='gray')
    axes[1].imshow(masks, cmap='gray', interpolation='nearest')
    axes[2].imshow(img, cmap='gray')
    axes[0].set_title("Original Atg8a channel")
    axes[1].set_title("Ground truth")
    axes[2].set_title("Prediction")

    plt.savefig(f'../plots/patches/{img_id}_predictions.png')
    plt.close()

def plot_prominence_vs_elevation(df, img_id):
    """Plot prominence vs elevation."""
    plt.figure()
    true_df = df[df['correct'] == 1]
    false_df = df[df['correct'] == 0]
    plt.plot(true_df['z_dist'], true_df['base_level'], 'r.', markersize=4)
    plt.plot(false_df['z_dist'], false_df['base_level'], 'b.', markersize=4)
    plt.xlabel('z_dist')
    plt.ylabel('base_level')
    plt.grid()
    plt.savefig(f'../plots/patches/{img_id}_prom_vs_el.png')
    plt.close()

def arrow_plot(image, df, img_id):
    masks_uint8 = cv2.imread(f'../../../data/patch_256_8bit/test/masks/{img_id}_masks.png', cv2.IMREAD_UNCHANGED)
    masks = masks_uint8 / 255
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    axes[0].imshow(image, cmap = 'gray')
    axes[1].imshow(masks, cmap="gray")
    axes[2].imshow(image, cmap = 'gray')
  
    for index, row in df.iterrows():
        x = row['x']
        y = row['y']
        axes[0].annotate('', 
                    xy=(x, y),  # coordinates of point to annotate
                    xytext=(x-5, y),  # coordinates of text
                    arrowprops=dict(facecolor='red', shrink=0.01, width=1, headwidth=4))
    for index, row in df.iterrows():
        axes[2].plot(row['x'], row['y'], marker='.', markersize=3, c='r')

    plt.savefig(f'../plots/patches/{img_id}_arrows.png')

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <img_id>  [min_z_dist]")
        sys.exit(1)

    img_id = sys.argv[1]
    min_z_dist = sys.argv[2]

    df, masks, img = load_data(img_id, min_z_dist)
    precision, recall, tp, fn, fp, fp_list, masks, multi_peaks, num_features = evaluate_results(df, masks)

    plot_results(masks, img, fp_list, img_id)
    plot_predictions(img, df, img_id)
    plot_prominence_vs_elevation(df, img_id)
    arrow_plot(img, df, img_id)

    print(f'Precision proxy: {precision*100:.2f}%')
    print(f'Recall proxy: {recall*100:.2f}%')
    print(f'TP: {int(tp)}/{num_features}')
    print(f'FN: {int(fn)}/{num_features}')
    print(f'FP: {int(fp)}')
    print(f'Multi-peaks: {multi_peaks} out of {num_features} GT masks')

if __name__ == "__main__":
    main()
