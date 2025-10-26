import sys
import numpy as np
import pandas as pd
import os
from osgeo import gdal, osr
from IPython import embed

def transform_coordinates(df, img_size, x_num, y_num):
    """
    Transform coordinates by rotating, flipping, and shifting.
    Assumes the long lat coordinates can be directly translated
    to x,y coordinates (neglect curvature of earth).
    
    Args:
    df (pd.DataFrame): Input DataFrame with 'lat' and 'long' columns.
    img_size (int): Size of the image patch.
    x_num (int): X-coordinate of the patch in the full image.
    y_num (int): Y-coordinate of the patch in the full image.
    
    Returns:
    pd.DataFrame: DataFrame with transformed 'x' and 'y' coordinates.
    """
    rotation_matrix = np.array([[0, 1], [-1, 0]])
    
    df_new = df.copy()
    df_new.rename(columns={'lat': 'x', 'long': 'y'}, inplace=True)
    
    for index, row in df.iterrows():
        point = np.array([row['lat'], row['long']]) * img_size
        # Rotate points 90 degrees counterclockwise
        point = np.dot(point - img_size // 2, rotation_matrix.T) + img_size // 2
        # Flip points vertically
        x, y = point
        y = point[1] - img_size
        # Shift point to correct place in full 2048*2048 image
        x = x + img_size * x_num
        y = y + img_size * y_num
        
        df_new.loc[index, 'x'] = x
        df_new.loc[index, 'y'] = y
    
    return df_new

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_id>")
        sys.exit(1)
    
    img_id = sys.argv[1]
    old_file = 'prominence/results.txt'
    img_size = 512  # Image patch size (should be parameterized in future)
    
    # Extract patch numbers from image ID
    x_num = int(img_id[-1])
    y_num = int(img_id[-4])
    
    # Read and process the input file
    df_old = pd.read_csv(old_file, delimiter=',', header=None)
    columns = ['lat', 'long', 'elevation', 'key_s_lat', 'key_s_long', 'prominence']
    df_old.columns = columns
    
    # Transform coordinates
    df_new = transform_coordinates(df_old, img_size, x_num, y_num)
    
    # Handle full-size results
    fullsize_id = img_id[:-6]
    file_path = f'../results/results_fullsize/{fullsize_id}.txt'
    
    if os.path.exists(file_path):
        df_fullsize = pd.read_csv(file_path, sep=',')
        df_new = pd.concat([df_fullsize, df_new], ignore_index=True)
    
    df_new.to_csv(file_path, sep=',', index=False)
    print(f"Results saved to {file_path}")

if __name__ == "__main__":
    main()
