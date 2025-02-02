import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import pandas as pd
import cv2

def tiff_to_array(tiff_path):
    with tifffile.TiffFile(tiff_path) as tiff:
        image = tiff.asarray() 
    return image

# def crop(image, start_x, start_y, crop_size):
#     crop_start_x = start_x
#     crop_start_y = start_y
#     image_cropped = image[crop_start_x:crop_start_x+crop_size,crop_start_y:crop_start_y+crop_size]
#     return image_cropped

def normalize_img(img):
    max_value = img.max() # normalize with recpect to max value
    min_value = img.min() # clip values below min
    image = (img-min_value)/max_value
    return image

def convert_to_8bit(image_16):
    max_value = image_16.max() # normalize with recpect to max value
    min_value = image_16.min() # clip values below min
    image_8 = ((image_16-min_value)/max_value*255).astype('uint8')
    return image_8

def plot_prediction(image, peak_dict,mask):
    plt.rcParams['figure.figsize'] = [15, 5] 
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    vmax = image.max()*0.8 # To brighten for better visabilty
    axes[0].imshow(image, cmap = 'gray', vmax = vmax)
    axes[1].imshow(mask, cmap="gray",interpolation='nearest')
    axes[2].imshow(image, cmap = 'gray', vmax = vmax)
    for key, value in peak_dict.items():
        x =  value['centroid'][0]
        y =  value['centroid'][1]
        axes[1].scatter(x, y, color= 'r', s = 6)
        axes[2].scatter(x, y, color= 'r', s = 6)

    
    axes[0].set_title("Original Atg8a channel")
    axes[1].set_title("Ground truth + predicted peaks")
    axes[2].set_title("Predicted peaks")
    
    plt.show()

def arrow_plot(image, masks, peak_dict):
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    axes[0].imshow(image, cmap = 'gray')
    axes[1].imshow(masks, cmap="gray")
    axes[2].imshow(image, cmap = 'gray')
  
    for key, value in peak_dict.items():
        axes[0].annotate('Point of interest', 
                    xy=(value['x'], value['y']),  # coordinates of point to annotate
                    xytext=(3, 3.5),  # coordinates of text
                    arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8))

    plt.show()

def evaluate_results(peak_dict, masks):
    """Evaluate results and calculate metrics."""
    masks_eval = masks.copy()
    df = pd.DataFrame.from_dict(peak_dict, orient='index')
    fp = len(df)
    fp_list = []
    #df['correct'] = 0

    for index, row in df.iterrows():
        x = int(row['centroid'][1])
        y = int(row['centroid'][0])
        df.at[index, 'x'] = x
        df.at[index, 'y'] = y
        gt_value = masks[x,y]
 
        fp -= gt_value
        df.at[index, 'correct'] = int(gt_value)
        if gt_value == 0:
            fp_list.append([x, y])

    ann_map, num_features = label(masks)
    tp, fn, multi_peaks = 0, 0, 0

    for value in range(1, num_features + 1):
        coords = np.argwhere(ann_map == value)
        coords_df = pd.DataFrame(coords, columns=['x', 'y'])
        merged_df = pd.merge(df, coords_df, on=['x', 'y'], how='inner')

        if len(merged_df) < 1:
            fn += 1
            masks_eval[coords[:, 0], coords[:, 1]] = 2
        else:
            tp += 1
            masks_eval[coords[:, 0], coords[:, 1]] = 1
            if len(merged_df) > 1:
                multi_peaks += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    tot = num_features
    print(f'Precision: {precision}\nRecall: {recall}\nTP:{tp}/{tot}\nFN:{fn}/{tot}\nFP:{fp}')
    return precision, recall, tp, fn, fp, masks_eval