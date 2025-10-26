import sys
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label


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