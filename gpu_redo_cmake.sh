#!/bin/bash/

rm -fr build/
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=icpx -DENABLE_CUDA=on # on is default
