#!/bin/bash

# Script to process an image using prominence analysis
# This script performs preprocessing, runs prominence analysis, and visualizes results

# Configuration
img_id="MF_MaxIP_3ch_2_000_230623_544_84_F_XY4_x2_y2"
unprocessed_img="../data/images_patch/${img_id}.tif"
processed_img="../data/images_georef/${img_id}_georef.tif"
min_elevation=200

# Function to run a command and wait for it to complete
run_command() {
    echo "Running: $1"
    $1
    wait
}

# Clean up existing prominence directory
if [ -d "prominence" ]; then
    echo "Removing existing prominence directory"
    rm -rf "prominence"
fi

# Preprocess the image
run_command "python3 preprocess.py \"${unprocessed_img}\""

# Run prominence analysis
run_command "python3 ../codes_prominence/scripts/run_prominence.py \
    --binary_dir ../codes_prominence/code/release \
    --threads 6 \
    --degrees_per_tile 1 \
    --skip_boundary \
    --min_prominence 150 \
    \"${processed_img}\""

# Convert results table
run_command "python3 convert_table.py \"${img_id}\""

# Visualize results
run_command "python3 visualize_results.py \"${img_id}\" $min_elevation"

# Analyze performance
run_command "python3 performance.py \"${img_id}\" $min_elevation"

echo "Processing complete for ${img_id}"
