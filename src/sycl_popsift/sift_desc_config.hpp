#pragma once

#include <sycl/sycl.hpp>

#if USE_JOINT_MATRIX
using FeatureType = sycl::half;
#else
using FeatureType = float;
#endif
