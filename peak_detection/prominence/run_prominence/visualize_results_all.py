import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from IPython import embed
from scipy.ndimage import rotate
import sys
import matplotlib.pyplot as plt
import numpy as np
# import gdal


# Load image
# image_id = 'MF_MaxIP_3ch_2_000_230623_544_84_F_XY4'
image_id =sys.argv[1]
min_elevation = sys.argv[2]
min_prominence = sys.argv[3]

img = cv2.imread(f'fullsize_images/images/{image_id}.tif', cv2.IMREAD_UNCHANGED)
masks = cv2.imread(f'fullsize_images/masks/{image_id}_masks.png', cv2.IMREAD_UNCHANGED)


# Load results
# Path to your text file
file_path = f'results_fullsize/{image_id}.txt'

# Read the text file into a DataFrame
df = pd.read_csv(file_path, delimiter=',')
filtered_df = df[df['elevation'] >= int(min_elevation)]
filtered_df = filtered_df[filtered_df['prominence'] >= int(min_prominence)]
df = filtered_df

# make plots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8), layout="constrained")
max_val = img.max() * 0.5
im1 = axes[0].imshow(img, cmap='gray', vmax = max_val)

# img_size = x_scale
for index, row in df.iterrows():
    axes[0].plot(row['x'], row['y'], marker='.', markersize=1, c='r')
im2 = axes[2].imshow(masks, cmap='gray', interpolation='nearest')
im3 = axes[1].imshow(img, cmap='gray', vmax = max_val)

fig.colorbar(im3, ax=axes.ravel().tolist(),location='right')


plt.savefig(f'plots/{image_id}.png')

plt.figure(figsize=(20, 20))
plt.imshow(img, cmap='gray', vmax = max_val)
for index, row in df.iterrows():
    plt.plot(row['x'], row['y'], marker='.', markersize=1, c='r')
plt.savefig(f'plots/{image_id}_large.png')
plt.clf()

plt.figure(figsize=(20, 20))
plt.imshow(img, cmap='gray', vmax = max_val)
plt.savefig(f'plots/{image_id}_large_original.png')