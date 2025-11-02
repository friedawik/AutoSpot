import cv2
import numpy as np
import pandas as pd
from IPython import embed
import matplotlib.pyplot as plt

""" This code performs watershed algorithm with peaks as seeds"""

def load_results(image_id):
    """Load and filter results from a CSV file."""
    file_path = f'../peak_detection/results/results_patch/{image_id}.txt'
    df = pd.read_csv(file_path)  
    return df

def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Compute the gradient magnitude of an image using the Sobel operator.
    
    Parameters:
        image (np.ndarray): Input image (should be in grayscale).
        
    Returns:
        np.ndarray: Gradient magnitude image.
    """
    # Ensure the image is in grayscale
    if len(image.shape) == 3:  # Check if it's a color image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Compute gradients using Sobel operator
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  # Gradient in x direction
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  # Gradient in y direction

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Convert magnitude to uint8 for visualization or further processing
    # Normalize the gradient magnitude to the range [0, 255]
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gradient_magnitude


# img_id = 'MF_MaxIP_3ch_2_000_230623_544_84_R1h_XY5_x0_y0'
# img_id='MF_MaxIP_3ch_2_000_230623_544_84_S4h_XY2_x1_y0'
img_id='MF_MaxIP_3ch_2_000_230623_544_84_S4h_XY2_x3_y3'

img_path = f'../data/patch_256_sam/test/images/{img_id}.png'
mask_path = f'../data/patch_256_sam/test/masks/{img_id}_masks.png'
gt_path =f'yolo/gt_plots/{img_id}.png'

# Step 1: Read the original TIFF image and the mask
gt_temp = cv2.imread(gt_path)
gt = cv2.cvtColor(gt_temp, cv2.COLOR_BGR2RGB)
original_image = cv2.imread(img_path)
grad_img = original_image.copy()
mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
df=load_results(img_id)

# Step 2: Preprocess the mask
# Ensure the mask is binary (0s and 255s)
_, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

three_channel_image = cv2.cvtColor(original_image[:,:,1], cv2.COLOR_GRAY2BGR)
three_channel_binary = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

mask_img = three_channel_image*three_channel_binary

# Step 3: Create an empty sure foreground mask
sure_fg = np.zeros_like(binary_mask, dtype=np.uint8)

# Step 4: Define sure foreground points (manual input)
# Convert columns to int
df['x'] = df['x'].astype(int)
df['y'] = df['y'].astype(int)
sure_foreground_points = list(zip(df['y'], df['x']))


# Set the pixels at these points to 255 (white) in sure_fg
for point in sure_foreground_points:
    y, x = point
    if binary_mask[y,x] != 0:
        if 0 <= y < sure_fg.shape[0] and 0 <= x < sure_fg.shape[1]:  # Check bounds
            sure_fg[y, x] = 255

# Step 5: Extract the unknown region
unknown = cv2.subtract(binary_mask, sure_fg)

# Marker labelling
num_labels, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))

# Increase marker values by 1 for algorithm
markers = markers + 1
markers[unknown == 255] = 0  # Mark the unknown region with zero

# Perform watershed
markers_1 = cv2.watershed(mask_img, markers)
# markers_2 = cv2.watershed(mask_img, markers)
original_image[markers_1 == -1] = [255, 0, 0]  # Mark watershed boundaries in red
original_image[sure_fg == 255] = [0, 255, 0]
mask_img[markers_1 == -1] = [0, 0, 255]  # Mark watershed boundaries in red
mask_img[sure_fg == 255] = [0, 0, 255]
# Step 4: Save the results
cropped_watershed_image = original_image[1:-1, 1:-1, :]
cv2.imwrite(f'{img_id}_watershed.png', cropped_watershed_image)
print("Results saved as 'output_watershed_result.png' and 'output_binary_mask.png'")


# Example usage:
# image = cv2.imread('path_to_your_image.png')
grad_magnitude = compute_gradient_magnitude(grad_img)
cv2.imwrite('output_gradient_magnitude.png', grad_magnitude)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), layout="constrained")
for ax in axes:
    ax.axis('off') 

cropped_image = gt[1:-1, 1:-1, :]
axes[0].imshow(cropped_image, cmap='gray')
axes[0].set_title('Ground Truth')


axes[1].imshow(cropped_watershed_image, cmap='gray', interpolation='nearest')
axes[1].set_title('Watershed Regions')

grad_magnitude = grad_magnitude[1:-1, 1:-1]
axes[2].imshow(grad_magnitude, cmap='gray')
axes[2].set_title('Gradient Magnitude')

plt.savefig(f'plots/watershed_{img_id}.png')