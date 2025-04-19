#!/bin/bash/

rm -fr build_no_experimental/
mkdir build_no_experimental && cd build_no_experimental 
cmake .. -DCMAKE_CXX_COMPILER=icpx -DENABLE_CUDA=on -DPopSift_EXPERIMENTAL=off
