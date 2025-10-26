# import sys
# import pandas as pd
# import cv2
# from IPython import embed
# from scipy.ndimage import label
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap


# img_id = sys.argv[1]
# min_elevation = sys.argv[2]
# if len(sys.argv) > 3:
#     max_prominence = sys.argv[3]
# else:
#     max_prominence = None  

# # Load results and GT mask
# df = pd.read_csv(f'../results/results_patch/{img_id}.txt', delimiter=',')
# df[['x', 'y']] = df[['x', 'y']].astype(int)
# masks_uint8 = cv2.imread(f'../data/masks/{img_id}_masks.png', cv2.IMREAD_UNCHANGED)
# masks = masks_uint8/255

# # filter low elevation
# filtered_df = df[df['elevation'] >= int(min_elevation)]
# if max_prominence is not None:
#     filtered_df = filtered_df[filtered_df['prominence'] >= int(max_prominence)]

# df = filtered_df.copy()

# # Get false positive by lopping through all peaks found by prominence and check the corresponding mask value

# fp = len(df) # initiate false positives as all points
# fp_list = []
# df['correct'] = 0

# for index, row in df.iterrows():
#     gt_value = masks[int(row['y']), int(row['x'])]
#     #tp += gt_value # add if inside mask
#     fp = fp - gt_value # subtract value 1 if inside mask 
#     if gt_value == 1:
       
#         df.at[index, 'correct'] = 1
#     if gt_value == 0:
#         fp_list.append([int(row['x']), int(row['y'])])
#         df.at[index, 'correct'] = 0


# # # Test with cv2 contours but some masks disappear 
# # contours, _ = cv2.findContours(masks_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # fn = 0
# # multi_peaks = 0
# # for contour in contours:
# #     peak_count = 0
# #     for index, row in df.iterrows():
# #         # +1, -1 or 0 depending on the point lying inside, outside or on the contour respectively
# #         if cv2.pointPolygonTest(contour,(row['x'], row['y']),False) >= 0:
# #             peak_count += 1

# #     if peak_count<1:
# #         fn += 1
# #     if peak_count>1:
# #         multi_peaks+=1
# #         print(peak_count)


# # Get mask count
# ann_map, num_features = label(masks)

# tp = 0  # True positives are masks that have one or more peaks found by prominence
# fn = 0  # False negatives are masks that have no peak in them

# small_gt_masks = 0
# multi_peaks = 0 # Count masks that have more than 1 peaks in them
# for value in range(1,num_features+1):
#     coords = np.argwhere(ann_map == value)
#     # if len(coords)<8:
#     #     small_gt_masks = small_gt_masks + 1
#     #     embed()
#     #     # plt.plot(coords[0][0], coords[0][1], markersize=6)
#     #     continue

#     # Convert the array to a DataFrame for easier comparison
#     # coords_df = pd.DataFrame(coords, columns=['x', 'y'])
#     coords_df = pd.DataFrame(coords, columns=['y', 'x'])
#     # Merge to find matching coordinates
#     merged_df = pd.merge(df, coords_df, on=['x', 'y'], how='inner')
#     # merged_df = pd.merge(df, coords_df, on=['y', 'x'], how='inner')

#     if len(merged_df)<1:
#         fn = fn + 1
#         masks[coords[:, 0], coords[:, 1]] = 2
#     elif len(merged_df)==1:
#         tp = tp + 1
#         multi_peaks = multi_peaks + 1
#         masks[coords[:, 0], coords[:, 1]] = 1
#     else:
#         tp = tp + 1
#         masks[coords[:, 0], coords[:, 1]] = 1


# precision = tp / (tp + fp)
# recall = tp / (tp+fn)  


# # Create a custom colormap
# colors = ['black', 'yellow', 'red']
# cmap = ListedColormap(colors)

# plt.figure()
# plt.imshow(masks, cmap=cmap, interpolation='nearest')
# # plt.imshow(ann_map)
# # for index, row in df.iterrows():
# #     plt.plot(row['x'], row['y'], marker='.', markersize=6, c='r')
# for row in fp_list:
#     plt.plot(row[0], row[1], marker='.', markersize=6, c='b')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.savefig(f'../plots/{img_id}_eval.png')

# print(f'precision proxy: {precision*100}\nrecall proxy: {recall*100}')
# print(f'tp: {int(tp)}/{value-small_gt_masks}')
# print(f'fn: {int(fn)}/{value-small_gt_masks}')
# print(f'fp: {int(fp)}')

# plt.cla()
# plt.figure()
# true_df = df[df['correct'] == 1]
# false_df = df[df['correct'] == 0]
# for index, row in true_df.iterrows():
#     plt.plot(row['elevation'], row['prominence'], marker='.', markersize=4, c='r')
# for index, row in false_df.iterrows():
#     plt.plot(row['elevation'], row['prominence'], marker='.', markersize=4, c='b')

# plt.xlabel('elevation')
# plt.ylabel('prominence')
# plt.grid()
# plt.savefig(f'../plots/{img_id}_prom_vs_el.png')
# # print(f'{multi_peaks} out of {len(contours)} gt masks had more than one peak')
# # print(f'masks found with cv2: {len(contours)}\nmasks found with scimage: {num_features} ')


import sys
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label

def load_data(img_id, min_elevation, max_prominence=None):
    """Load and filter data."""
    df = pd.read_csv(f'../results/results_patch/{img_id}.txt', delimiter=',')
    df[['x', 'y']] = df[['x', 'y']].astype(int)
    masks_uint8 = cv2.imread(f'../data/masks/{img_id}_masks.png', cv2.IMREAD_UNCHANGED)
    masks = masks_uint8 / 255

    df = df[df['elevation'] >= int(min_elevation)]
    if max_prominence is not None:
        df = df[df['prominence'] >= int(max_prominence)]

    return df, masks

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
    plt.plot(true_df['elevation'], true_df['prominence'], 'r.', markersize=4)
    plt.plot(false_df['elevation'], false_df['prominence'], 'b.', markersize=4)
    plt.xlabel('elevation')
    plt.ylabel('prominence')
    plt.grid()
    plt.savefig(f'../plots/{img_id}_prom_vs_el.png')
    plt.close()

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <img_id> <min_elevation> [max_prominence]")
        sys.exit(1)

    img_id = sys.argv[1]
    min_elevation = sys.argv[2]
    max_prominence = sys.argv[3] if len(sys.argv) > 3 else None

    df, masks = load_data(img_id, min_elevation, max_prominence)
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
