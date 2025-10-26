#import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def tiff_to_array(tiff_path):
    with tifffile.TiffFile(tiff_path) as tiff:
        image = tiff.asarray() 
    return image


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

def plot_prediction(image, prediction ,mask, bbox_list, input_patch, img_id):
    plt.rcParams['figure.figsize'] = [15, 15] 
    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
    vmax = image.max()*0.8 # To brighten for better visabilty
    axes[0][0].imshow(image, cmap = 'gray', vmax = vmax)
    axes[0][1].imshow(mask, cmap="gray",interpolation='nearest')
    axes[1][0].imshow(input_patch, cmap = 'gray')
    axes[1][1].imshow(prediction, cmap = 'gray')

    for bbox in bbox_list:
        #x_min, y_min, width, height = bbox
        # x_min, y_min, x_max, y_max = bbox
        y_min, x_min, y_max, x_max = bbox
        width = x_max-x_min
        height = y_max-y_min
        rect1 = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        axes[0][0].add_patch(rect1)
        axes[1][1].add_patch(rect2)
    axes[0][0].set_title("Original Atg8a channel with bbox inputs")
    axes[0][1].set_title("Ground truth")
    axes[1][0].set_title("Contour masks")
    axes[1][1].set_title("Predicted peaks")
    fig.suptitle(f'{img_id}', fontsize=16)
    plt.savefig('test.png')