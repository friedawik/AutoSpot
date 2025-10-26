## Peak Detection Using Mountains Code 

This repository contains code to apply Andrew Kirmse's "mountains" algorithm on fluorescent microscopy data to detect ATG8A peaks in grayscale images.
### Installation of 'Mountains'
To get started, you will need to install the "mountains" source code, which can be found at the following URL: https://github.com/akirmse/mountains.

### Preparing Images
Before running the analysis, it's essential to prepare your images. The original images (2048x2048 pixels) are too large for processing, so they need to be divided into smaller patches. Each image patch will be converted into a georeferenced .tif file, as the mountains code is designed to work with geographic data. When preparing these patches, ensure that they are scaled appropriately so that each patch corresponds to a small geographic distance, allowing for the neglect of Earth's curvature. This script preprocess.py handles these tasks.

### Running the Analysis
The run_prominence folder contains scripts that facilitate the analysis of both image patches and full-sized images. For full-sized images, the mountains code first processes the smaller patch images and then reassembles the resukts into their original size for further analysis. The scripts run_prominence_fullsize.sh and run_prominence_patch.sh handle these tasks.

### Visualization and Performance
The Python scripts included in this repository assist with visualizing and evaluating the performance of the peak detection process.

### Further reading
More information about the "mountains" code can be found at: https://www.andrewkirmse.com/andrew-kirmses-home-page
