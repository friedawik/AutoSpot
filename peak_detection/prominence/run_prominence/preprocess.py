from osgeo import gdal, osr
import numpy as np
import sys
import os
from IPython import embed
import rasterio

"""
Pre-processing to convert a tif image to a georeferenced tif images with 
long lat coordinates instead of Cartesian coordinates. Each Cartesian patch 
is defined as a one-degree patch in longitude and latitude.
"""


# Open the unreferenced TIFF file
input_tif = sys.argv[1]
output_tif = sys.argv[2]

directory, file_name = os.path.split(input_tif)
name, ext = os.path.splitext(file_name)
new_name =  output_tif
dataset = gdal.Open(input_tif)

# Get the dimensions of the image
width = dataset.RasterXSize
height = dataset.RasterYSize

# Define arbitrary scaled dimensions (adjust as needed)
scaled_width = 1
scaled_height = 1

# Calculate the scaling factors
x_scale = scaled_width / width
y_scale = scaled_height / height

# Create a new geotransform
geotransform = [0, x_scale, 0, 0, 0, -y_scale]

# Create a new spatial reference system (using WGS84 as an example)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

# Create the output georeferenced TIFF
driver = gdal.GetDriverByName("GTiff")
out_dataset = driver.Create(new_name, width, height, dataset.RasterCount, dataset.GetRasterBand(1).DataType)

# Set the geotransform and projection
out_dataset.SetGeoTransform(geotransform)
out_dataset.SetProjection(srs.ExportToWkt())

# Copy the data from the input to the output
for i in range(1, dataset.RasterCount + 1):
    in_band = dataset.GetRasterBand(i)
    out_band = out_dataset.GetRasterBand(i)
    out_band.WriteArray(in_band.ReadAsArray())

# Close the datasets
dataset = None
out_dataset = None

print(f"Georeferenced TIFF saved as {new_name}")
