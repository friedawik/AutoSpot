#!/bin/bash

if [ -d "prominence" ]; then
    rm -rf "prominence"
fi
if [ -d "tiles" ]; then
    rm -rf "tiles"
fi

# Increase the argument limit for prompt. I had some issues with length limitation running the data merge step in prominence.
ulimit -s 600000

img_id="MF_MaxIP_3ch_2_000_230623_544_84_F_XY4_x0_y2"

INPUT="../../data/patch_256_fed/test/images/${img_id}.tif"
OUTPUT="../data/images_georef/${img_id}_georef.tif"

echo "Preprocessing image $INPUT"
gdal_translate -b 2 -of GTiff -ot Float32 "$INPUT" temp.tif
python preprocess.py temp.tif "$OUTPUT"



# Execute the script
python ../../tools/mountains/scripts/run_prominence.py   \
      --binary_dir ../../tools/mountains/code/release \
      --threads 18  \
      --degrees_per_tile 1 \
      --samples_per_tile 256 \
      --skip_boundary \
      --min_prominence 120 \
      "$OUTPUT"

min_elevation=180
python3 convert_table_patch.py "${img_id}"
python3 visualize_results_patch.py "${img_id}" $min_elevation
python3 performance_patch.py "${img_id}" $min_elevation


