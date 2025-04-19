#!/bin/bash

cd ./build_no_experimental/

# If make works it returns 0 and hence exit wont run
# due to short circuting
make || exit 1
cd ./Linux-x86_64/

# time ./popsift-demo --input-file ~/Downloads/sample_1280×853.pgm
time ./popsift-demo --input-file ~/Downloads/sample_640×426.pgm

