import tifffile
import numpy as np

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