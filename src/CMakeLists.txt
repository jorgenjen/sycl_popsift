add_library(
  sycl_popsift
  sycl_popsift/popsift.cpp
  sycl_popsift/popsift.hpp
  sycl_popsift/s_image.cpp
  sycl_popsift/s_image.hpp
  sycl_popsift/sift_pyramid.cpp
  sycl_popsift/sift_pyramid.hpp
  sycl_popsift/s_pyramid_build.cpp
  sycl_popsift/s_pyramid_build_ra.cpp
  sycl_popsift/s_pyramid_build_aa.cpp
  sycl_popsift/sift_constants.cpp
  sycl_popsift/sift_constants.hpp
  sycl_popsift/sift_octave.cpp
  sycl_popsift/sift_octave.hpp
  sycl_popsift/gauss_filter.cpp
  sycl_popsift/gauss_filter.hpp
  sycl_popsift/s_extrema.cpp
  sycl_popsift/sift_extremum.h
  sycl_popsift/s_orientation.cpp
  sycl_popsift/features.cpp
  sycl_popsift/features.hpp
  # sycl_popsift/s_desc_loop.cpp // moved into desc
  # sycl_popsift/s_desc_loop.hpp // moved into desc
  sycl_popsift/sift_desc.cpp
  sycl_popsift/non_sycl/sift_conf.cpp
  sycl_popsift/non_sycl/sift_conf.hpp
  sycl_popsift/common/subgroup_bitonic_sort.hpp
  sycl_popsift/common/debug_macros.cpp
  sycl_popsift/common/debug_macros.hpp
  sycl_popsift/common/sync_queue.h
  sycl_popsift/common/assist.h)

# TODO: Figure out if it should be STATIC, SHARED or not set like in popsift
# (aka using BUILD_SHARED_LIBS to do do it differntly for MSVC and shared libs

# add_sycl_to_target(TARGET sycl_popsift SOURCES sycl_popsift/popsift.cpp
# sycl_popsift/popsift.hpp sycl_popsift/s_image.cpp sycl_popsift/s_image.hpp
# sycl_popsift/sift_pyramid.cpp sycl_popsift/sift_pyramid.hpp
# sycl_popsift/s_pyramid_build.cpp sycl_popsift/s_pyramid_build_ra.cpp
# sycl_popsift/s_pyramid_build_aa.cpp sycl_popsift/sift_constants.cpp
# sycl_popsift/sift_constants.hpp sycl_popsift/sift_octave.cpp
# sycl_popsift/sift_octave.hpp sycl_popsift/gauss_filter.cpp
# sycl_popsift/gauss_filter.cpp sycl_popsift/s_extrema.cpp
# sycl_popsift/s_orientation.cpp sycl_popsift/malloc_devt.hpp
# sycl_popsift/common/subgroup_bitonic_sort.hpp )

# GPU target_compile_options(sycl_popsift PUBLIC ${SYCL_FLAGS})
# target_link_libraries(sycl_popsift PUBLIC Threads::Threads ${SYCL_FLAGS} )
target_compile_options(sycl_popsift PUBLIC ${SYCL_FLAGS} ${CPU_ONLY})
target_link_libraries(sycl_popsift PUBLIC ${SYCL_FLAGS})

# CPU target_link_libraries(sycl_popsift PUBLIC Threads::Threads )

# CPU target_link_libraries(sycl_popsift PUBLIC Threads::Threads )

# build directory containing the automatically generated files
set(popsift_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# BUILD_INTERFACE allows to include the directory with source only when target
# is built in the building tree (ie, not from an install location)
target_include_directories(
  sycl_popsift
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
         $<BUILD_INTERFACE:${popsift_generated_dir}>
         $<BUILD_INTERFACE:${popsift_generated_dir}/sycl_popsift>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>)

# TODO: Add install target like popsift

if(PopSift_BUILD_EXAMPLES)
  message(STATUS "Building demo!")
  add_subdirectory(application)
else()
  message(STATUS "Not building demo!")
endif()
