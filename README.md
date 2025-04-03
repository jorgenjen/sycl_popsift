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

- [ ] Improve error handlng (preferably asynchronous errors (refer to chap 5 in book)). 

- [ ] Make it work for cpu aswell (could not work due to local and global sizes does not match with what the cpu can handle? device_multiple? Look into using prefered device multiple instead to create the nd_ranges for the kernels

- [ ] Get rid of unnecessary shared USM memory (hapers performance)


## Profiling/analysis

Easy way to profile individual kernels and operations that is tied to an event by comparing event start and end. From sycl documentation

```c++
#include <sycl/sycl.hpp>

#include <cstdlib>
#include <cstring>


int main() {
  sycl::property_list properties{sycl::property::queue::enable_profiling()};
  auto q = sycl::queue(sycl::default_selector_v, properties);

  std::cout
      << "  Platform: "
      << q.get_device().get_platform().get_info<sycl::info::platform::name>()
      << std::endl;

  const int num_ints = 1024 * 1024;
  const size_t num_bytes = num_ints * sizeof(int);
  const int alignment = 8;

  // Alloc memory on host
  auto src = std::aligned_alloc(alignment, num_bytes);
  std::memset(src, 1, num_bytes);

  // Alloc memory on device
  auto dst = sycl::malloc_device<int>(num_ints, q);
  q.memset(dst, 0, num_bytes).wait();

  // Copy from host to device
  auto event = q.memcpy(dst, src, num_bytes);
  event.wait();


  auto end =
      event.get_profiling_info<sycl::info::event_profiling::command_end>();

  auto start =
      event.get_profiling_info<sycl::info::event_profiling::command_start>();

  std::cout << "Elapsed time: " << (end - start) / 1.0e9 << " seconds\n";

  sycl::free(dst, q);
}
```
