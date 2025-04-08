#!/bin/bash/

rm -fr build_cpu/
mkdir build_cpu && cd build_cpu
cmake .. -DCMAKE_CXX_COMPILER=icpx -DENABLE_CUDA=off



