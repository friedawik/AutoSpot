#!/bin/bash

# Loop through all .tif files in the images_patch directory
for file in ../../data/patch_256_starved/test/images/*.tif; do
    # Extract the image ID from the filename (remove .tif extension)
    img_id=$(basename "$file" .tif)

    echo "Processing image: $img_id"
    echo "Processing image: $file"

    # Remove existing prominence directory if it exists
    if [ -d "prominence" ]; then
        echo "Removing existing prominence directory"
        rm -rf "prominence"
    fi

    INPUT=$file
    OUTPUT="../data/images_georef/${img_id}_georef.tif"

    # Preprocess the image
    echo "Preprocessing image $INPUT"
    gdal_translate -b 2 -of GTiff -ot Float32 "$INPUT" temp.tif
    python preprocess.py temp.tif "$OUTPUT"
    # python3 preprocess.py "${unprocessed_img}"
    
    # Run prominence analysis
    echo "Running prominence analysis"
    python3 ../../tools/mountains/scripts/run_prominence.py  \
        --binary_dir ../../tools/mountains/code/release \
        --threads 1  \
        --degrees_per_tile 1 \
        --samples_per_tile 256 \
        --skip_boundary \
        --min_prominence 180 \
        "$OUTPUT"

    # Convert table to full size
    echo "Converting table to full size"
    python3 convert_table_fullsize.py "${img_id}"
    python3 convert_table_patch.py "${img_id}"

    echo "Finished processing $img_id"
    echo "------------------------"
done

echo "All images processed"