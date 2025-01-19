import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from IPython import embed

# Function to calculate the distance between two points
def calculate_distance(x0, y0, x1, y1):
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

# Function to calculate the bounding box
def bounding_box(x0, y0, radius):
    # Calculate the bounding box coordinates
    left = x0 - radius
    right = x0 + radius
    top = y0 - radius
    bottom = y0 + radius

    return (left, top, right, bottom)

# Load image
img = cv2.imread('test.tiff', cv2.IMREAD_UNCHANGED)
fig, ax = plt.subplots()
plt.imshow(img, cmap='gray')

# Rereference
x_scale = img.shape[0]
y_scale = img.shape[0]

# Define the transformation matrices
rotation_matrix = np.array([[0, 1],
                            [-1,  0]])

flip_matrix = np.array([[-1, 0],
                        [ 0, 1]])

# Load results
file_path = 'prominence/results.txt'

# Read the text file into a DataFrame
df = pd.read_csv(file_path, delimiter=',', header=None)

# Assign appropriate column names if desired
df.columns = ['lat', 'long', 'elevation', 'key_s_lat', 'key_s_long', 'prominence']

lim = 60
img_size = x_scale
for index, row in df.iterrows():
    if row['prominence'] > lim:
        #  re-reference points. FIXME: y should not need to be rescaled
        point = np.array([row['lat'], row['long']])*x_scale
        point = np.dot(point - img_size // 2, rotation_matrix.T) + img_size // 2
        x, y = point
        y = point[1]-img_size
        # print(f'x={x} y={y}')
        # get sadle point
        sadle_point = np.array([row['key_s_lat'], row['key_s_long']])*x_scale
        sadle_point = np.dot(sadle_point - img_size // 2, rotation_matrix.T) + img_size // 2
        s_x, s_y = sadle_point
        s_y = sadle_point[1]-img_size
        # print(f'x={s_x} y={s_y}')
        
        plt.plot(x, y, marker='.', markersize=2)
        radius = calculate_distance(x, y, s_x, s_y)
        if radius<20:
            print(radius)
        # print(radius)
        # Get the bounding box
            bbox = bounding_box(x, y, radius)
            rect = plt.Rectangle((bbox[0], bbox[1]), 2*radius, 2*radius, edgecolor='red', facecolor='none')
            # Plot the circle and bounding box
            ax.add_patch(rect)
          
      

plt.savefig('bbox.png')