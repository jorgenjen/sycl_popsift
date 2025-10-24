#!/bin/bash

# Setup: adjust these paths to your actual install locations
export ONEAPI_ROOT=/home/jorgen/intel/oneapi
export PATH=$ONEAPI_ROOT/compiler/2025.0/bin:$PATH
export LD_LIBRARY_PATH=$ONEAPI_ROOT/compiler/2025.0/lib:$LD_LIBRARY_PATH

# Codeplay plugin: ensure any required CUDA backend libs are included
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Optional: print for debugging
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "Launching $@"

# Run the application
exec "$@"

