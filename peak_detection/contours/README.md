## Peak detection using multiple contour levels
The matplotlib.pyplot.contour function is used to extract contours at multiple levels. Any overlapping are used to build a feature which can be included or excluded based on the following parameters:
- pixel value difference between base contour and peak contour 
- area 
- circularity
- peak pixel value

### Installation
Install the required libraries with conda using: 
conda env create -f environment.yml -n new_environment_name

### Run codes
The scripts in the folder codes can be used to run the method on a set of images. For this repository, image patches of 256x256 pixels were used and the results were added together to perform analysis on the original 2048x2048 pixel images. 

### Results

|Image  |Precision (proxy)|Recall|Accuracy|F1  |FP  |TP  |FN |
|-------|-----------------|------|--------|----|----|----|---|
|Fed    |54.3             |82.6  |N/A     |65.5|340 |404 |85 |
|Starved|70.5             |91.5  |N/A     |79.7|639 |1527|141|
|Refed  |94.0             |86.4  |N/A     |90.1|127 |1999|314|
|Total  |78.0             |87.9  |N/A     |82.7|1106|3930|540|

The results were produces with the following parameters:
min_area = 3
max_area = 200
levels = 200
min_z_dist = 100

A true positive peak is defined as a peak placed inside a GT mask. If there are several peaks in a mask, all will be counted as true positives (thereof the 'proxy' in precision). If a peak is outside a mask, it will be counted as a false negative. 
### Notes: should be tested with overlap or bigger patches, loosing quite some peaks