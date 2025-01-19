import sys
import numpy as np
import pandas as pd

def transform_coordinates(df, img_size):
    """
    Transform coordinates by rotating and flipping.
    
    Args:
    df (pd.DataFrame): Input DataFrame with 'lat' and 'long' columns.
    img_size (int): Size of the image.
    
    Returns:
    pd.DataFrame: DataFrame with transformed 'x' and 'y' coordinates.
    """
    # Define transformation matrices
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
        
        df_new.loc[index, 'x'] = x
        df_new.loc[index, 'y'] = y
    
    return df_new

def main():
    # Check if correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_id>")
        sys.exit(1)
    
    img_id = sys.argv[1]
    old_file = '../run_prominence/prominence/results.txt'
    img_size = 512  # Image size (should be parameterized in future)
    
    # Read the input file
    df_old = pd.read_csv(old_file, delimiter=',', header=None)
    columns = ['lat', 'long', 'elevation', 'key_s_lat', 'key_s_long', 'prominence']
    df_old.columns = columns
    
    # Transform coordinates
    df_new = transform_coordinates(df_old, img_size)
    
    # Save transformed results
    output_file = f'../results/results_patch/{img_id}.txt'
    df_new.to_csv(output_file, sep=',', index=False)
    print(f"Transformed results saved to {output_file}")

if __name__ == "__main__":
    main()

