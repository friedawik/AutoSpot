#!/bin/bash

if [ -d "prominence" ]; then
    rm -rf "prominence"
fi

# Wait for all background processes to finish
wait

img_id="MF_MaxIP_3ch_2_000_230623_544_84_F_XY4_x2_y2"
unprocessed_img="../data/images_patch/${img_id}.tif"
processed_img="../data/images_georef/${img_id}_georef.tif"


python3 preprocess.py "${unprocessed_img}"

python3 ../codes_prominence/scripts/run_prominence.py  \
      --binary_dir ../codes_prominence/code/release \
      --threads 6  \
      --degrees_per_tile 1 \
      --skip_boundary \
      --min_prominence 150 \
      "${processed_img}"

# Wait for all background processes to finish
wait

# mv prominence/results.txt results/$img_id.txt
min_elevation=200
python3 convert_table_patch.py "${img_id}"
python3 visualize_results_patch.py "${img_id}" $min_elevation
wait
python3 performance_patch.py "${img_id}" $min_elevation
