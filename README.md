## Changes in structure from Popsift CUDA

1. Move d_gauss from global to an attribute of popsift. As you canot pass global variables to a sycl kernel. And sycl does not support constant memory (global memory with better caching) as cuda does
2. Modify grid_divide so that it gives the complete outer size as sycl does not do blockidx like cuda.

