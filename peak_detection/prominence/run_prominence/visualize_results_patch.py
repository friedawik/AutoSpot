# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import cv2
# from IPython import embed
# from scipy.ndimage import rotate
# import sys
# import matplotlib.pyplot as plt
# import numpy as np



# # Load image

# image_id =sys.argv[1]
# elevation_min = sys.argv[2]

# img = cv2.imread(f'../data/images_georef/{image_id}_georef.tif', cv2.IMREAD_UNCHANGED)
# masks = cv2.imread(f'../data/masks/{image_id}_masks.png', cv2.IMREAD_UNCHANGED)
# # img = cv2.imread('test.tiff', cv2.IMREAD_UNCHANGED)
# # get patch number
# # x_num = image_id[-4]
# # y_num = image_id[-1]

# # img = cv2.imread('test_images/MF_MaxIP_3ch_2_000_230623_544_84_F_XY4.tiff', cv2.IMREAD_UNCHANGED)
# # rotated_img = rotate(img, 90)
# # vertically_flipped_image = np.flipud(rotated_img)
# # plt.imshow(vertically_flipped_image, cmap='gray')

# # Scale points back to original img
# x_scale = img.shape[0]
# y_scale = img.shape[0]
# # geotransform = [0, x_scale, 0, 0, 0, -y_scale]
# # success, inv_geotransform = gdal.InvGeoTransform(geotransform)

# # Load results
# # Path to your text file
# file_path = f'../results/results_patch/{image_id}.txt'

# # Read the text file into a DataFrame
# # Adjust the delimiter parameter as needed (e.g., '\t' for tab, ',' for comma, ' ' for space)
# # df = pd.read_csv(file_path, delimiter=',', header=None)
# df = pd.read_csv(file_path, delimiter=',')
# filtered_df = df[df['elevation'] >= int(elevation_min)]
# df = filtered_df

# # make plots
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), layout="constrained")
# max_val = img.max() * 0.6
# im1 = axes[0].imshow(img, cmap='gray', vmax = max_val)

# img_size = x_scale
# for index, row in df.iterrows():
#     axes[0].plot(row['x'], row['y'], marker='.', markersize=6, c='r')
# im2 = axes[1].imshow(masks, cmap='gray', interpolation='nearest')
# im3 = axes[2].imshow(img, cmap='gray', vmax = max_val)

# fig.colorbar(im3, ax=axes.ravel().tolist(),location='right')


# # plt.tight_layout()

# plt.savefig(f'../plots/{image_id}.png')

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

def load_image(image_id, image_type):
    """Load an image file."""
    if image_type == 'georef':
        return cv2.imread(f'../data/images_georef/{image_id}_georef.tif', cv2.IMREAD_UNCHANGED)
    elif image_type == 'mask':
        return cv2.imread(f'../data/masks/{image_id}_masks.png', cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError("Invalid image type")

def load_results(image_id, elevation_min):
    """Load and filter results from a CSV file."""
    file_path = f'../results/results_patch/{image_id}.txt'
    df = pd.read_csv(file_path)
    return df[df['elevation'] >= int(elevation_min)]

def plot_results(img, masks, df, image_id):
    """Create and save plots of the results."""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), layout="constrained")
    
    max_val = img.max() * 0.6
    
    # Plot original image with points
    im1 = axes[0].imshow(img, cmap='gray', vmax=max_val)
    for _, row in df.iterrows():
        axes[0].plot(row['x'], row['y'], marker='.', markersize=6, c='r')
    
    # Plot masks
    im2 = axes[1].imshow(masks, cmap='gray', interpolation='nearest')
    
    # Plot original image again
    im3 = axes[2].imshow(img, cmap='gray', vmax=max_val)
    
    # Add colorbar
    fig.colorbar(im3, ax=axes.ravel().tolist(), location='right')
    
    # Save the figure
    plt.savefig(f'../plots/{image_id}.png')

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_id> <elevation_min>")
        sys.exit(1)

    image_id = sys.argv[1]
    elevation_min = sys.argv[2]

    # Load images
    img = load_image(image_id, 'georef')
    masks = load_image(image_id, 'mask')

    # Load and filter results
    df = load_results(image_id, elevation_min)

    # Create and save plots
    plot_results(img, masks, df, image_id)

if __name__ == "__main__":
    main()
