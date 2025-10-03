if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  # I am top-level project, i.e. I am not being include by another project
  # cmake_minimum_required(VERSION 3.12)
  message(STATUS "Applicaton building as top-level project")
  # TODO: Test that this actually works when run as top level

  # Followring documentation
  if(CMAKE_HOST_WIN32)
    # need CMake 3.25.0+ for IntelLLVM support of target link properties on
    # Windows
    cmake_minimum_required(VERSION 3.25)
    include(CMakeForceCompiler)
    cmake_force_cxx_compiler(icx-cl IntelDPCPP) # Should probably change this to
                                                # IntelSYCL if possible
    include(Platform/Windows-Clang)
  else()
    # CMake 3.23.5 is the minimum recommended for IntelLLVM on Linux
    cmake_minimum_required(VERSION 3.23.5)
    set(CMAKE_CXX_COMPILER icpx)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g") # To avod optimizations and
                                                     # enable debugging symbols
                                                     # in addition to line below
                                                     # making debugging easier
  endif()

  project(PopsiftDemo LANGUAGES CXX)

  # Recommended over IntelDPCPP according to intel docs it might be deprecated
  # in the future
  find_package(IntelSYCL REQUIRED) # should de added after project definition

  option(PopSift_BOOST_USE_STATIC_LIBS
         "Link examples with static Boost libraries" OFF)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)

  include(GNUInstallDirs)

  set(CMAKE_POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
      "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
      "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(TARGET sycl_popsift)
  # when compiled in the repository the target is already defined
  add_library(PopSift::popsift ALIAS sycl_popsift)
else()
  # Add NO_CMAKE_BUILDS_PATH for windows if using CMake-GUI to build packages to
  # avoid searching in temporary build directory of Foo project See 5: *
  # http://www.cmake.org/cmake/help/v3.0/command/find_package.html

  # This requires the install to work correctly of the library
  find_package(PopSift CONFIG REQUIRED)
  message(STATUS "Found PopSift, version: ${PopSift_VERSION}")
endif()

find_package(DevIL COMPONENTS IL ILU) # yields IL_FOUND, IL_LIBRARIES,
                                      # IL_INCLUDE_DIR

if(PopSift_BOOST_USE_STATIC_LIBS)
  set(Boost_USE_STATIC_LIBS ON)
endif()

# TODO: Not sure if I need system for sycl_popsift

# Seems to be fine when using Boost 1.7 or newer to supress the warning However
# seems like when minimum cmake is lower than 3.30 (as is the case here) you
# need the OLD Policy that uses FindBoost.cmake and not BostConfig.cmake TODO:
# Figure out if this is sfine or if I need to increase minimum cmake or use OLD
# or another solution
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

# Removing .0
find_package(Boost 1.71 REQUIRED COMPONENTS filesystem program_options system)

# find_package(Boost 1.71.0 REQUIRED COMPONENTS filesystem program_options
# system)
if(WIN32)
  add_definitions("-DBOOST_ALL_NO_LIB")
endif(WIN32)

set(PD_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
# set(PD_LINK_LIBS       ${Boost_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY})
set(PD_LINK_LIBS ${Boost_LIBRARIES})

if(IL_FOUND OR DevIL_FOUND)
  message(STATUS "DevIL found")
  set(PD_COMPILE_OPTIONS "-DUSE_DEVIL")
  list(APPEND PD_INCLUDE_DIRS ${IL_INCLUDE_DIR})
  list(APPEND PD_LINK_LIBS ${IL_LIBRARIES} ${ILU_LIBRARIES})
else()
  message(WARNING "DevIL not found -- Falling back to pgmread")
  set(PD_COMPILE_OPTIONS "")
endif()

# ##############################################################################
# popsift-demo
# ##############################################################################

add_executable(popsift-demo main.cpp pgmread.cpp pgmread.h)

# set_property(TARGET popsift-demo PROPERTY CXX_STANDARD 11)

# add_sycl_to_target(TARGET popsift-demo SOURCES main.cpp pgmread.cpp pgmread.h)

target_compile_options(popsift-demo PRIVATE ${PD_COMPILE_OPTIONS})
target_include_directories(popsift-demo PUBLIC PopSift::popsift
                                               ${PD_INCLUDE_DIRS})
target_compile_definitions(popsift-demo PRIVATE ${Boost_DEFINITIONS})
target_link_libraries(popsift-demo PUBLIC PopSift::popsift ${PD_LINK_LIBS})

# ##############################################################################
# popsift-match
# ##############################################################################

add_executable(popsift-match match.cpp pgmread.cpp pgmread.h)
#
# set_property(TARGET popsift-match PROPERTY CXX_STANDARD 11)
#
target_compile_options(popsift-match PRIVATE ${PD_COMPILE_OPTIONS})
target_include_directories(popsift-match PUBLIC PopSift::popsift
                                                ${PD_INCLUDE_DIRS})
target_compile_definitions(popsift-match PRIVATE ${Boost_DEFINITIONS})
target_link_libraries(popsift-match PUBLIC PopSift::popsift ${PD_LINK_LIBS})

# ##############################################################################
# popsift-perf-test --> Executables that sample data
# ##############################################################################

add_executable(popsift-perf-test matching_performance_test.cpp pgmread.cpp
                                 pgmread.h)

target_compile_options(popsift-perf-test PRIVATE ${PD_COMPILE_OPTIONS})
target_include_directories(popsift-perf-test PUBLIC PopSift::popsift
                                                    ${PD_INCLUDE_DIRS})
target_compile_definitions(popsift-perf-test PRIVATE ${Boost_DEFINITIONS})
target_link_libraries(popsift-perf-test PUBLIC PopSift::popsift ${PD_LINK_LIBS})

# ##############################################################################
# Running a whole dataset with warmup
# ##############################################################################

add_executable(popsift-frametime-test frame_time_perf_test.cpp pgmread.cpp
                                      pgmread.h)

target_compile_options(popsift-frametime-test PRIVATE ${PD_COMPILE_OPTIONS})
target_include_directories(popsift-frametime-test PUBLIC PopSift::popsift
                                                         ${PD_INCLUDE_DIRS})

target_compile_definitions(popsift-frametime-test PRIVATE ${Boost_DEFINITIONS})
target_link_libraries(popsift-frametime-test PUBLIC PopSift::popsift
                                                    ${PD_LINK_LIBS})

# ##############################################################################
# Retriving large pool of descriptors
# ##############################################################################

add_executable(popsift-extract-descriptors
               performance_tests/retrive_descriptors.cpp pgmread.cpp pgmread.h)

target_compile_options(popsift-extract-descriptors
                       PRIVATE ${PD_COMPILE_OPTIONS})

target_include_directories(popsift-extract-descriptors
                           PUBLIC PopSift::popsift ${PD_INCLUDE_DIRS})

target_compile_definitions(popsift-extract-descriptors
                           PRIVATE ${Boost_DEFINITIONS})
target_link_libraries(popsift-extract-descriptors PUBLIC PopSift::popsift
                                                         ${PD_LINK_LIBS})

# ##############################################################################
# Performance testing of mattching using pool extracted above ^
# ##############################################################################

add_executable(
  popsift-matc-perf-test-pool
  performance_tests/performance_testing_of_matching.cpp pgmread.cpp pgmread.h)

target_compile_options(popsift-matc-perf-test-pool
                       PRIVATE ${PD_COMPILE_OPTIONS})

target_include_directories(popsift-matc-perf-test-pool
                           PUBLIC PopSift::popsift ${PD_INCLUDE_DIRS})

target_compile_definitions(popsift-matc-perf-test-pool
                           PRIVATE ${Boost_DEFINITIONS})
target_link_libraries(popsift-matc-perf-test-pool PUBLIC PopSift::popsift
                                                         ${PD_LINK_LIBS})

# ##############################################################################
# installation
# ##############################################################################

install(TARGETS popsift-demo DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS popsift-match DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS popsift-perf-test DESTINATION ${CMAKE_INSTALL_BINDIR})
