#!/bin/bash/

rm -fr build/
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2025.0/bin/icpx -DENABLE_CUDA=off



