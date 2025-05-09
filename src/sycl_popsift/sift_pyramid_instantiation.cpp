// #include "sift_pyramid.hpp"
// #include "pyramid_build.cpp"  // Pull in definitions
// #include "pyramid_extrema.cpp" // (Optional, see Alternative below)

#include "sift_pyramid.hpp"
#include "sycl_popsift/sift_desc.cpp"

// Include all cpp files that is a part of the Pyramid class
// This allows the linker to have everything before instantiation
#include "sycl_popsift/s_extrema.cpp"
#include "sycl_popsift/s_orientation.cpp"
#include "sycl_popsift/s_pyramid_build.cpp"
#include "sycl_popsift/s_pyramid_build_aa.cpp"
#include "sycl_popsift/sift_pyramid.cpp"

namespace popsift {

template class Pyramid<float>;
template class Pyramid<sycl::half>;

} // namespace popsift
