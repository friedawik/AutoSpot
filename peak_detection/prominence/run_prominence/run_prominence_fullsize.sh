# #!/bin/bash

# for file in ../data/images_patch/*; do
#     img_id=$(basename "$file" .tif)

#     if [ -d "prominence" ]; then
#         rm -rf "prominence"
#     fi

#     # Wait for all background processes to finish
#     wait

#     unprocessed_img="../data/images_patch/${img_id}.tif"
#     processed_img="../data/images_georef/${img_id}_georef.tif"


#     python3 preprocess.py "${unprocessed_img}"
    
#     python3 ../codes_prominence/scripts/run_prominence.py  \
#         --binary_dir ../codes_prominence/code/release \
#         --threads 6  \
#         --degrees_per_tile 1 \
#         --skip_boundary \
#         --min_prominence 120 \
#         "${processed_img}"

#     python3 convert_table_fullsize.py "${img_id}"

# done

#!/bin/bash

# Script to process multiple image files using prominence analysis

# Loop through all .tif files in the images_patch directory
for file in ../data/images_patch/*.tif; do
    # Extract the image ID from the filename (remove .tif extension)
    img_id=$(basename "$file" .tif)

    echo "Processing image: $img_id"

    # Remove existing prominence directory if it exists
    if [ -d "prominence" ]; then
        echo "Removing existing prominence directory"
        rm -rf "prominence"
    fi

    # Define input and output image paths
    unprocessed_img="../data/images_patch/${img_id}.tif"
    processed_img="../data/images_georef/${img_id}_georef.tif"

    # Preprocess the image
    echo "Preprocessing image"
    python3 preprocess.py "${unprocessed_img}"
    
    # Run prominence analysis
    echo "Running prominence analysis"
    python3 ../codes_prominence/scripts/run_prominence.py  \
        --binary_dir ../codes_prominence/code/release \
        --threads 6  \
        --degrees_per_tile 1 \
        --skip_boundary \
        --min_prominence 120 \
        "${processed_img}"

    # Convert table to full size
    echo "Converting table to full size"
    python3 convert_table_fullsize.py "${img_id}"

    echo "Finished processing $img_id"
    echo "------------------------"
done

echo "All images processed"
