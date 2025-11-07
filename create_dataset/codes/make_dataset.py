import numpy as np
import os
import tifffile 
from IPython import embed
from cellpose import models, plot, utils, io
from cellpose.io import imread
import cv2
from sklearn.model_selection import train_test_split

# read three channel tif images and convert to np array
def tiff_to_array(tiff_path):
    with tifffile.TiffFile(tiff_path) as tiff:
        image = tiff.asarray() 
    return image

# Convert to 8 bit if needed, could expand to include clipping if needed
def convert_to_8bit(image_16):
    max_value = image_16.max() 
    min_value = image_16.min() 
    image_8 = ((image_16-min_value)/(max_value-min_value)*255).astype('uint8')
    return image_8


# Patch parameters
size_x = 512
size_y = 512

new_folder = f'patch_{size_x}_single'

# Get GT images for train and validation dataset
cwd = os.getcwd()
folder_path = cwd + "/full_size/train_val/" 

# Loop through masks and correponding images
for file_name in os.listdir(folder_path):
    if file_name[-4:] == '.tif':
        continue
    elif file_name[-4:] == '.npy':
        image_name = file_name[:-8]
        image_path = folder_path + image_name + '.tif'
        image_16bit = tiff_to_array(image_path)
        # image = convert_to_8bit(image_16bit)
    
        num_patch = image_16bit.shape[0]//size_x
        new_size = num_patch*size_x
        image = image_16bit[:new_size,:new_size,:]
        # image = image[:,:,1] # get atg8a channel
        # image = image_16bit
        seg_path = folder_path + file_name
        seg_object = np.load(seg_path, allow_pickle=True).item()
        
        # make binary
        binary_mask = seg_object['masks']
        binary_mask[binary_mask > 0] = 255
        binary_mask = binary_mask.astype(np.uint8)

        # Loop with sliding window to make patches
        image_dict = {}
        index = 0
        count_y = 0
        for y in range(0, image.shape[0], size_x):
            count_x = 0
            for x in range(0, image.shape[1], size_y):
                img_patch = image[x:x + size_x , y:y + size_y][:,:,1]
                img_patch_id = image_name + f'_x{count_x}_y{count_y}.tif'
                # img_patch_id = image_name + f'_x{count_x}_y{count_y}.png'
                mask_patch = binary_mask[x:x + size_x , y:y + size_y]
                mask_patch_id = image_name + f'_x{count_x}_y{count_y}_masks.png'
                image_dict[index] = {'mask' : mask_patch, 
                                    'mask_id' : mask_patch_id,  
                                    'image' : img_patch,
                                    'image_id': img_patch_id 
                                    }
                index +=1
                count_x += 1
            count_y += 1

        # Split the data
        train, val = train_test_split(image_dict, test_size=0.2, random_state=42)
   
        # Loop through list to save train dataset
        for row in train:
            img_patch = row['image']
            img_patch = convert_to_8bit(img_patch)
            mask_patch = row['mask']
            
            mask_dir = os.path.join(new_folder,'train/masks/')
            img_dir = os.path.join(new_folder,'train/images/')
            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(img_dir, exist_ok=True)
            patch_mask_path = os.path.join(mask_dir,row['mask_id'])
            patch_img_path = os.path.join(img_dir, row['image_id'])
            tifffile.imwrite(patch_img_path, img_patch.astype(np.uint16))   # save as 16 bit tif file
            cv2.imwrite(patch_mask_path, mask_patch)                        # save as 8 bit png file
            # cv2.imwrite(patch_img_path, img_patch)  
        print(f"Made {len(train)} train patches of shape {img_patch.shape} of image {image_name}.")

        # Loop through list to save validation dataset
        for row in val:
            img_patch = row['image']
            img_patch = convert_to_8bit(img_patch)
            mask_patch = row['mask']

            # Make folders
            mask_dir = os.path.join(new_folder,'val/masks/')
            img_dir = os.path.join(new_folder,'val/images/')
            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(img_dir, exist_ok=True)
            patch_mask_path = os.path.join(mask_dir,row['mask_id'])
            patch_img_path = os.path.join(img_dir, row['image_id'])

            tifffile.imwrite(patch_img_path, img_patch.astype(np.uint16))   # save as 16 bit tif file
            cv2.imwrite(patch_mask_path, mask_patch)                        # save as 8 bit png file
            # cv2.imwrite(patch_img_path, img_patch)  
        print(f"Made {len(val)} val patches of shape {img_patch.shape} of image {image_name}.")

# Get GT images for train and validation dataset
folder_path = cwd + "/full_size/test/" 

# Loop through full size masks and corresponding images 
for file_name in os.listdir(folder_path):
    if file_name[-4:] == '.tif':
        continue
    elif file_name[-4:] == '.npy':
        image_name = file_name[:-8]
        image_path = folder_path + image_name + '.tif'
        image_16bit = tiff_to_array(image_path)
        # image = convert_to_8bit(image_16bit)
        num_patch = image_16bit.shape[0]//size_x
        new_size = num_patch*size_x
        image = image_16bit[:new_size,:new_size,:]
        # image = image[:,:,1] # get atg8a channel
        seg_path = folder_path + file_name
        seg_object = np.load(seg_path, allow_pickle=True).item()
        seg = seg_object['masks']
 
        #make binary
        binary_mask = seg
        binary_mask[binary_mask > 0] = 255
        binary_mask = binary_mask.astype(np.uint8)

        # Make patches and save them directly as tif and png
        index = 0
        count_y = 0
        for y in range(0, image.shape[0], size_x):
            count_x = 0
            for x in range(0, image.shape[1], size_y):
                img_patch = image[x:x + size_x , y:y + size_y][:,:,1]
                # img_patch = convert_to_8bit(img_patch)

                # Make folders
                mask_dir = os.path.join(new_folder,'test/masks/')
                img_dir = os.path.join(new_folder,'test/images/')
                os.makedirs(mask_dir, exist_ok=True)
                os.makedirs(img_dir, exist_ok=True)
                patch_mask_path = os.path.join(mask_dir,image_name + f'_x{count_x}_y{count_y}_masks.png')
                patch_img_path = os.path.join(img_dir, image_name + f'_x{count_x}_y{count_y}.tif')
          


                # patch_img_path = 'patch_256_8bit/test/images/' + image_name + f'_x{count_x}_y{count_y}.tif'
                # patch_img_path = 'patch_256_8bit/test/images/' + image_name + f'_x{count_x}_y{count_y}.png'
                mask_patch = binary_mask[x:x + size_x , y:y + size_y]
                # patch_mask_path = 'patch_256_8bit/test/masks/' + image_name + f'_x{count_x}_y{count_y}_masks.png'

                tifffile.imwrite(patch_img_path, img_patch.astype(np.uint16))   # save as 16 bit tif file
                cv2.imwrite(patch_mask_path, mask_patch)                        # save as 8 bit png file
                # cv2.imwrite(patch_img_path, img_patch) 

                index +=1
                count_x += 1
            count_y += 1
        print(f"Made {index} test patches of shape {img_patch.shape} of image {image_name}.")
