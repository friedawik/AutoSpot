import numpy as np
import os
import tifffile 
from IPython import embed
import matplotlib.pyplot as plt

def tiff_to_array(tiff_path):
    with tifffile.TiffFile(tiff_path) as tiff:
        image = tiff.asarray() 
    return image

def convert_to_8bit(image_16):
    # Ensure the input is a 3D array
    if image_16.ndim != 3:
        raise ValueError("Input must be a 3D array")

    # Initialize an empty array for the result
    image_8 = np.empty_like(image_16, dtype='uint8')

    # Normalize each channel independently
    for i in range(image_16.shape[2]):
        channel = image_16[:,:,i]
        max_value = channel.max()
        min_value = channel.min()
        
        # Avoid division by zero
        if max_value == min_value:
            image_8[:,:,i] = np.zeros_like(channel, dtype='uint8')
        else:
            image_8[:,:,i] = ((channel - min_value) / (max_value - min_value) * 255).astype('uint8')

    return image_8

def normalize_channels(image_16):
    # Ensure the input is a 3D array
    if image_16.ndim != 3:
        raise ValueError("Input must be a 3D array")

    # Create an empty array for the result
    image_8 = np.empty_like(image_16, dtype='uint8')

    # Normalize each channel independently
    for i in range(image_16.shape[2]):
        cv2.normalize(image_16[:,:,i], image_8[:,:,i], 0, 255, cv2.NORM_MINMAX)

    return image_8

def swap_red_green(image):
    # Ensure the input is a 3D RGB image
    if image.shape[-1] != 3:
        raise ValueError("Input must be an RGB image with 3 channels.")
    
    # Swap the red and green channels
    swapped_image = image.copy()
    swapped_image[..., 0], swapped_image[..., 1] = image[..., 1], image[..., 0]
    
    return swapped_image


# Get test images paths 
folder_dir = os.path.join(os.getcwd(), 'full_size','test')
f_img_dir = os.path.join(folder_dir,'MF_MaxIP_3ch_2_000_230623_544_84_F_XY4.tif')
s4h_img_dir = os.path.join(folder_dir,'MF_MaxIP_3ch_2_000_230623_544_84_R1h_XY5.tif')
r1h_img_dir = os.path.join(folder_dir,'MF_MaxIP_3ch_2_000_230623_544_84_S4h_XY2.tif')

# Open and convert images to 8bit png for visualisation in thesis

f_16bit = tiff_to_array(f_img_dir)
f_8bit = convert_to_8bit(f_16bit)
f_plot = swap_red_green(f_8bit)

# f_8bit = normalize_channels(f_img)

s4h_16bit = tiff_to_array(s4h_img_dir)
s4h_8bit = convert_to_8bit(s4h_16bit)
s4h_plot = swap_red_green(s4h_8bit)

r1h_16bit = tiff_to_array(r1h_img_dir)
r1h_8bit = convert_to_8bit(r1h_16bit)
r1h_plot = swap_red_green(r1h_8bit)


# Create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
plt.tight_layout()

for ax in [ax1, ax2, ax3]:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

ax1.imshow(f_plot)
ax1.set_title('Fed')

ax2.imshow(s4h_plot)
ax2.set_title('Starved')

ax3.imshow(r1h_plot)
ax3.set_title('Refed')


# Plot circ vs area
plt.savefig(f'plots/fullsize_8bit.png')
plt.cla()


# Create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
plt.tight_layout()

for ax in [ax1, ax2, ax3]:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

ax1.imshow(f_plot[:,:,0], cmap='gray')
ax1.set_title('Fed')

ax2.imshow(s4h_plot[:,:,0], cmap='gray')
ax2.set_title('Starved')

ax3.imshow(r1h_plot[:,:,0], cmap='gray')
ax3.set_title('Refed')


# Plot circ vs area
plt.savefig(f'plots/fullsize_grayscale.png')
plt.cla()