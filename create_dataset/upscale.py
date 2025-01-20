import os
import cv2
from IPython import embed
import numpy as np

# New dimension
new_size = 512
# Path to data folder
data_dir = os.path.join(os.getcwd(), 'patch_256_8bit')
new_data_dir = os.path.join(os.getcwd(), 'patch_512')

# Loop through all dataset folders and open each mask image. 
# Get the contours and save in a txt file following yolo format
count = 0
for item in os.listdir(data_dir):
    folder_dir = os.path.join(data_dir,item)
    # make new folders
    imgs_new_dir = os.path.join(new_data_dir,item,'images')
    os.makedirs(imgs_new_dir, exist_ok=True)
    masks_new_dir = os.path.join(new_data_dir,item,'masks')
    os.makedirs(masks_new_dir, exist_ok=True)
    labels_new_dir = os.path.join(new_data_dir,item,'labels')
    os.makedirs(labels_new_dir, exist_ok=True)


    if os.path.isdir(folder_dir):
        masks_dir = os.path.join(folder_dir, 'masks')
        for mask_image in os.listdir(masks_dir):
            # make paths     

            # Read image, resize and save
    
            img_dir = os.path.join(folder_dir,'images',f'{mask_image[:-10]}.png')
            image = cv2.imread(img_dir)
            resized_image = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
            img_new_dir = os.path.join(imgs_new_dir,f'{mask_image[:-10]}.png')
         
            cv2.imwrite(img_new_dir, resized_image)

            # Read mask image, resize and save
            mask_dir = os.path.join(masks_dir,mask_image)
            mask = cv2.imread(mask_dir)
            resized_mask = cv2.resize(mask, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
            mask_new_dir = os.path.join(masks_new_dir,mask_image)
            cv2.imwrite(mask_new_dir, resized_mask)

            # Get contours and save them in txt file
            label_file = os.path.join(labels_new_dir, f'{mask_image[:-10]}.txt')

            mask_shape = resized_mask.shape[0]  # can only handle squared patches


            contours, _ = cv2.findContours(resized_mask[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Flatten contours and save in txt file
            # os.makedirs(labels_dir, exist_ok=True)
       
            with open(label_file, 'w') as f:
                for coords in contours:
                    flat_coords = coords.flatten()
                    norm_coords = flat_coords / mask_shape
                    full_line =  np.insert(norm_coords, 0, 0) # add 0 for spot class
                    if len(full_line) > 5:
                        coordinates_str = ' '.join(map(str, full_line))
                        f.write(f'{coordinates_str}\n')
                    else:
                        count += 1
                        print(f'{count} spots too small')

         