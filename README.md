## Changes in structure from Popsift CUDA

1. Move d_gauss from global to an attribute of popsift. As you canot pass global variables to a sycl kernel. And sycl does not support constant memory (global memory with better caching) as cuda does
2. Modify grid_divide so that it gives the complete outer size as sycl does not do blockidx like cuda.
3. Modify add another scale fucntion that is not bound to a class so we deal wit two widths heights
    - Done like this due to the way popsift utilizes the texture engine to do the upscale and I'm currently not using that so need to replicate teh same functionality in software

## Known problems currently:

1. Segfault happening when running in quick succession. Seems to happen on first octave after or during level 5 so could be related to level 5 (final level) or subsampling for next octave. Always happens for the first octave atelast in the examples I've gotten so far.
    - Now it seems to not segfault anymore but fails every other run that is back to back due to 

2. Doing kernel invocatons wrong in 2d as linearization should be as describled in documentation(see below) to get the sub-groups to read from coaleced memory to replicate the cuda code and get the sub-groups to equal the accesses of the cuda warps.
    - So in sycl we need the following conversions from cuda kernel to sycl in 2D
        - cuda.x --> nd[1] && cuda.y --> nd[0]
    - in 3D:
        - cuda.x --> nd[2] && cuda.y --> nd[1] && cuda.z --> nd[0]
        
![sycl linearization](/linearization_in_sycl.png)


## Things that needs to be done:

1. Improve error handlng (preferably asynchronous errors (refer to chap 5 in book)). 
