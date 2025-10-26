# import tifffile
import numpy as np
import matplotlib.pyplot as plt

# def tiff_to_array(tiff_path):
#     with tifffile.TiffFile(tiff_path) as tiff:
#         image = tiff.asarray() 
#     return image


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

def plot_prediction(image, prediction ,mask):
    plt.rcParams['figure.figsize'] = [15, 5] 
    fig, axes = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    vmax = image.max()*0.8 # To brighten for better visabilty
    axes[0].imshow(image, cmap = 'gray', vmax = vmax)
    axes[1].imshow(mask, cmap="gray",interpolation='nearest')
    axes[2].imshow(prediction, cmap = 'gray', vmax = vmax)

    
    axes[0].set_title("Original Atg8a channel")
    axes[1].set_title("Ground truth + predicted peaks")
    axes[2].set_title("Predicted peaks")
    
    plt.show()