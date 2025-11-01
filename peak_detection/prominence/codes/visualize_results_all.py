import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from IPython import embed
from scipy.ndimage import rotate
import sys
import matplotlib.pyplot as plt
import numpy as np

"""
This code visualizes the results obtained from the 'mountains' algorithm applied to the patches that collectively form a complete 2048x2048 image.

Args:
    image_id (str): The identifier for the image to be visualized, provided as a command-line argument.
    min_elevation (int): The minimum elevation threshold used for visualization, specified as a command-line argument.
    min_prominence (int): The minimum prominence value for features to be included in the visualization, also specified as a command-line argument.
"""

# Load image
image_id =sys.argv[1]
min_elevation = sys.argv[2]
min_prominence = sys.argv[3]

img = cv2.imread(f'../../data/full_size/test/{image_id}.tif', cv2.IMREAD_UNCHANGED)
masks = cv2.imread(f'../../data/full_size/test/masks/{image_id}_masks.png', cv2.IMREAD_UNCHANGED)

# Load results
file_path = f'../results/results_fullsize/{image_id}.txt'

# Read the text file into a DataFrame and filter on prominence, elevation
df = pd.read_csv(file_path, delimiter=',')
filtered_df = df[df['elevation'] >= int(min_elevation)]
filtered_df = filtered_df[filtered_df['prominence'] >= int(min_prominence)]
df = filtered_df

# make plot that shows peak detection results, original image, gt masks
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8), layout="constrained")
max_val = img.max() * 0.5
im1 = axes[0].imshow(img, cmap='gray', vmax = max_val)
for index, row in df.iterrows():
    axes[0].plot(row['x'], row['y'], marker='.', markersize=1, c='r')
im2 = axes[2].imshow(masks, cmap='gray', interpolation='nearest')
im3 = axes[1].imshow(img, cmap='gray', vmax = max_val)

fig.colorbar(im3, ax=axes.ravel().tolist(),location='right')
plt.savefig(f'../plots/{image_id}.png')

# make figure that shows all peaks in image
plt.figure(figsize=(20, 20))
plt.imshow(img, cmap='gray', vmax = max_val)
plt.axis('equal')  # Set equal scaling for both axes
plt.axis('off')  # Remove axes
for index, row in df.iterrows():
    plt.plot(row['x'], row['y'], marker='.', markersize=1, c='r')
plt.savefig(f'../plots/{image_id}_large.png', bbox_inches='tight')
plt.clf()
