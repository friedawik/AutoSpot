import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# get training image
image_name = f'MF_MaxIP_3ch_2_000_230623_544_84_F_XY4'
image_path = '../../../data/fullsize/test/images/' + image_name + '.png'
mask_path = '../../../data/fullsize/test/masks/' + image_name+ '_masks.png'

# Open images and gt masks
img = cv2.imread(image_path,  cv2.IMREAD_UNCHANGED)



plt.imshow(img, cmap='gray')
plt.savefig('test.png')


# get training image
image_name = f'MF_MaxIP_3ch_2_000_230623_544_84_F_XY4_x0_y7'
image_path = '../../../data/patch_256_8bit/test/images/' + image_name + '.png'
mask_path = '../../../data/patch_256_8bit/test/masks/' + image_name+ '_masks.png'

# Open images and gt masks
img = cv2.imread(image_path,  cv2.IMREAD_UNCHANGED)


fig = plt.figure(figsize=(20,20))
plt.imshow(img, cmap='gray')
plt.savefig('test_1.png')