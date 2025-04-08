#!/bin/bash

cd ./build_cpu/

# If make works it returns 0 and hence exit wont run
# due to short circuting
make || exit 1
cd ./Linux-x86_64/

./popsift-demo --input-file ../../sample_640×426.pgm

