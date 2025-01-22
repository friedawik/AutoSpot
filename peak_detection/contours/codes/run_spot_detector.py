import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import tifffile
#from skimage import data, feature, color
#from scipy.ndimage import grey_erosion, grey_dilation
#import plotly.graph_objects as go
import pandas as pd
#from segment_anything import SamPredictor, sam_model_registry
from tools import *
from spot_detection_tool import SpotDetector
from IPython import embed

# get training image
start_x = 2
start_y = 2
image_name = f'MF_MaxIP_3ch_2_000_230623_544_84_F_XY1_x{start_x}_y{start_y}'
image_path = '../../../data/patch_256/train/images/' + image_name + '.tif'
mask_path = '../../../data/patch_256/train/masks/' + image_name+ '_masks.png'

# Open images and gt masks
img = tiff_to_array(image_path)

img = img[:,:,1]
img = convert_to_8bit(img)
#plt.figure(figsize = (10,10))
#plt.imshow(img, cmap = 'gray')
#plt.savefig("../plots/original_patch.png")


masks = cv2.imread(mask_path)

# Set parameters for contours search
min_area= 3
max_area= 200
levels = 10
min_z = 25
spots=SpotDetector(img,contour_levels=levels, min_area=min_area, max_area=max_area,min_circularity = 0.3, z_min_dist = min_z)
all_dict=spots.get_features()
feature_dict = spots.check_z_distance(all_dict)

