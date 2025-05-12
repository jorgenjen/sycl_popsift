#!/bin/bash

cd ./build/

# If make works it returns 0 and hence exit wont run
# due to short circuting
# make || exit 1
cd ./Linux-x86_64/

time ./popsift-match -l ~/Downloads/AI_GENERATED_VS_REAL_DATASET_SAFE/0EAwg7WIIMhgnSfLf.png -r ~/Downloads/pgm_imgs/sample_1280×853.pgm
