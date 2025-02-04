## Changes in structure from Popsift CUDA

1. Move d_gauss from global to an attribute of popsift. As you canot pass global variables to a sycl kernel. And sycl does not support constant memory (global memory with better caching) as cuda does
2. Modify grid_divide so that it gives the complete outer size as sycl does not do blockidx like cuda.
3. Modify add another scale fucntion that is not bound to a class so we deal wit two widths heights
    - Done like this due to the way popsift utilizes the texture engine to do the upscale and I'm currently not using that so need to replicate teh same functionality in software

## Known problems currently:

1. Segfault happening when running in quick succession. Seems to happen on first octave after or during level 5 so could be related to level 5 (final level) or subsampling for next octave. Always happens for the first octave atelast in the examples I've gotten so far.


