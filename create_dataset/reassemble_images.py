import numpy as np

im1 = 'MF_MaxIP_3ch_2_000_230623_544_84_F_XY4'
im2 = 'MF_MaxIP_3ch_2_000_230623_544_84_R1h_XY5'
im3 = 'MF_MaxIP_3ch_2_000_230623_544_84_S4h_XY2'


full_img = np.zeros(2048, 2048)

for x in range(4):
    
    for y in range(4):