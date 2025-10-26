## Spot segmentation using multiple contour levels in combination with SAM2
The matplotlib.pyplot.contour function is used to extract contours at multiple levels. Any overlapping contours are used to build a feature which can be included or excluded based on the following parameters:
- pixel value difference between base contour and peak contour 
- area 
- circularity
- peak pixel value

The masks are used as inputs to the Segment Anything Model 2 (SAM2).

### Installation
Install the required libraries with conda using: 
conda env create -f environment.yml -n new_environment_name

### Run codes
The scripts in the folder codes can be used to run the method on a set of images. For this repository, image patches of 256x256 pixels were used and the results were added together to perform analysis on the original 2048x2048 pixel images. 

### Results

### Notes: should be tested with overlap or bigger patches, loosing quite some peaks