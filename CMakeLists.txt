cmake_minimum_required(VERSION 3.12)
project(sycl_popsift)

# if(CMAKE_BUILD_TYPE STREQUAL "") message(STATUS "CMAKE_BUILD_TYPE was not set,
# setting to Release") set(CMAKE_BUILD_TYPE "Release") endif()

set(CMAKE_CXX_STANDARD 17) # required for sycl

# # Require threads because of std::thread. find_package(Threads REQUIRED)

# TODO: Add more options like Popsift has if suitable
option(PopSift_BUILD_EXAMPLES "Build PopSift applications." ON)
# option(BUILD_SHARED_LIBS "Build shared libraries" ON)

option(PopSift_EXPERIMENTAL "Use experimantal features." ON) # bindless images
# These are toggles that only have effect whem PopSift_EXPERIMENTAl are ON
option(BindlessInput "Use bindless input image for upscaling" ON)

# Not fully implemented (hence should stay off, Uses USM when ON anyways)
option(BindlessArray "Use bindless image array for pyramid" ON)

# Set to off as default and should stay off until it has a working version
option(JointMatrix "Use Joint Matrix for matching (specific funtion)" OFF)

# option(JointMatrixBlur "Use Joint Matrix for building the image pyramid" OFF)

# This option is only to attempt to get better performance for the most time
# using part throught one long lasting wave instead of many kernel launches Only
# used for sufficiently large octaves
option(
  PersistentThreads
  "Use threads that last for the whole octave building, only used for sufficiently large octaves"
  OFF)

# For some performance analysis of multiple kernels as I've not been able to
# make nsight systems record kernel's, But for singular kernel lauches I've been
# able to use nsight compute so this flags does not do kernel by kernel
# timerecording as nsight compute deals with that (but could be added) --
# IMPORANT to remember that the profiling itself adds a small overhead (accurate
# due to using sycl queue profiler)
option(Profiling "Time certain parts of application" OFF)

option(
  PerfTestingFunctions
  "Compiles different functions used for performance testing and data generation"
  OFF)

# option(Profiling "For profiling mode (used for nsight)" ON)

# To generate compile_commands.json -- useful for clangd LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set build path as a folder named as the platform (linux, windows, darwin...)
# plus the processor type
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

# Configures gpu target for sycl automatically both nvidia, amd and spriv
include(cmake/ConfigureSYCL.cmake)

# For debug symbols remove for release

# set(CMAKE_BUILD_TYPE Debug) # Seems like this makes the kernel prefix have 66
# registers more than it can have max 64 for 32 x 32 dimensions (this is for gpu
# version) not sure if debug symbols or if it is the lower optimization level
# that causes the register count to be too high

# Should mby lower the y dimension prefix sum in debug mode if we want that to
# be compatible with the gpu  version

# # Add debug symbols and disable optimization set(CMAKE_CXX_FLAGS
# "${CMAKE_CXX_FLAGS} -g -O0")
#
# # Enable AddressSanitizer set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}
# -fsanitize=address")

# list(APPEND SYCL_FALGS -fsycl-device-code-split=per_kernel)

# TODO: Make it like this for amd and intel gpu's too so you can toggle on and
# off cpu only
if(${ENABLE_CUDA})
  message(STATUS "CUDA enabled")
  # set(CPU_ONLY "")
else()
  message(STATUS "CUDA DISABLED -- Running cpu only mode")
  # set(CPU_ONLY "-DCPU_ONLY")

  add_compile_definitions(CPU_ONLY=1)

  # set(CMAKE_BUILD_TYPE Debug) message(STATUS "Running in debug mode :D")
  # list(APPEND SYCL_FLAGS -O0 -g -fno-omit-frame-pointer)
  set(CMAKE_BUILD_TYPE Release)
  list(APPEND SYCL_FLAGS -O3)
endif()

if(${Profiling})
  # message(STATUS "Running with lineinfo flag fro profiling")
  message(
    STATUS
      "Profiling -- Compiling with --lineinfo and -q with -O3, Also using sycl queue profiling for overall metrics"
  )
  # list(APPEND SYCL_FLAGS -gline-tables-only -g -O3)
  list(APPEND SYCL_FLAGS -O3 -fno-fast-math)
  # list(APPEND SYCL_FLAGS -gline-tables-only)

  # For sycl queue profiling (for overall frame performance and such)
  add_compile_definitions(QUEUE_PROFILING=1)
endif()

if(${PersistentThreads})
  add_compile_definitions(USE_PERSISTENT=1)
endif()

#
# list(APPEND SYCL_FLAGS -lineinfo) endif()

# this if and above can be refactored to a better logic.. Currently not that
# readable
if(PopSift_EXPERIMENTAL AND ${ENABLE_CUDA})
  message(STATUS "Using experimantal features!")
  # set(EXPERIMENTAL "-DUSE_EXPERIMENTAL")
  if(BindlessInput)
    add_compile_definitions(USE_BINDLESS_INPUT=1)
  else()
    add_compile_definitions(USE_BINDLESS_INPUT=0)
  endif()

  if(BindlessArray)
    add_compile_definitions(USE_BINDLESS_ARRAY=1)
  else()
    add_compile_definitions(USE_BINDLESS_ARRAY=0)
  endif()

  # Not cuda exclusive
  if(JointMatrix)
    add_compile_definitions(USE_JOINT_MATRIX=1)
    message(STATUS "Joint matrix is ${JointMatrix} - Using fp16 for descriptor")
  else()
    add_compile_definitions(USE_JOINT_MATRIX=0)
    message(
      STATUS "Joint matrix is ${JointMatrix} - Using float for descriptor")
  endif()

  # Not using Joint matrix for blurint due to using fp16 and fp32 mix so not
  # well for repeated convolutions

  # if(JointMatrixBlur) add_compile_definitions(BLUR_JOINT_MATRIX=1) message(
  # STATUS "JointMatrixBlur ${JointMatrixBlur} -- Using Joint matrix to build
  # pyramid" ) else() add_compile_definitions(BLUR_JOINT_MATRIX=1) message(
  # STATUS "JointMatrixBlur ${JointMatrixBlur} -- Normal split convolution for
  # pyramid building" ) endif()
else()
  # Non CUDA or non Experimental
  if(PopSift_EXPERIMENTAL)
    message(STATUS "Using experimantal features!")
    # Enable none cuda exlusive extensions

    if(JointMatrix)
      add_compile_definitions(USE_JOINT_MATRIX=1)
      message(
        STATUS "Joint matrix is ${JointMatrix} - Using fp16 for descriptor")
    else()
      add_compile_definitions(USE_JOINT_MATRIX=0)
      message(
        STATUS "Joint matrix is ${JointMatrix} - Using float for descriptor")
    endif()
  else()
    message(STATUS "Building without experimental features")
    # Disable none cuda exclusive extensions
  endif()

  # set(EXPERIMENTAL "-DUSE_EXPERIMENTAL 0") set(EXPERIMENTAL "")

  # Currently bindless images are cuda exclusive
  add_compile_definitions(USE_BINDLESS_INPUT=0)
  add_compile_definitions(USE_BINDLESS_ARRAY=0)
  # Not cuda exclusive
  add_compile_definitions(USE_JOINT_MATRIX=0)

endif()

if(PerfTestingFunctions)
  add_compile_definitions(PERF_TESTING_FUNCTIONS=1)
  message(STATUS "Performance Testing functions ${PerfTestingFunctions}")
else()
  add_compile_definitions(PERF_TESTING_FUNCTIONS=0)
  message(STATUS "Performance Testing functions ${PerfTestingFunctions}")
endif()

# Need to figure out number of sub-groups per Compute Unit for GPU (this can be
# overriden in config class passed to PopSift)

if(${ENABLE_CUDA})
  message(STATUS "CUDA_COMPUTE_CAPABILITY = ${CUDA_COMPUTE_CAPABILITY}")
  add_compile_definitions(CUDA_CC=${CUDA_COMPUTE_CAPABILITY})
endif()

if(${ENABLE_HIP})
  add_compile_definitions(HIP_ARCH=${HIP_GFX_ARCH})
endif()

# ==============================================================================
# GNUInstallDirs CMake module - Define GNU standard installation directories -
# Provides install directory variables as defined by the GNU Coding Standards.
# ==============================================================================
include(GNUInstallDirs)

add_subdirectory(src)

# # Followring documentation if (CMAKE_HOST_WIN32) # need CMake 3.25.0+ for
# IntelLLVM support of target link properties on Windows
# cmake_minimum_required(VERSION 3.25) include (CMakeForceCompiler)
# CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP) # Should probably change this to
# IntelSYCL if possible include (Platform/Windows-Clang) else() # CMake 3.23.5
# is the minimum recommended for IntelLLVM on Linux
# cmake_minimum_required(VERSION 3.22.1) set(CMAKE_CXX_COMPILER icpx) endif()
#
# project(sycl_popSift VERSION 1.0.0 LANGUAGES CXX)
#
# # Recommended over IntelDPCPP according to intel docs it might be deprecated
# in the future find_package(IntelSYCL REQUIRED) # should de added after project
# definition # IntelSYCL -- should be  this but doeos not work need to fix
# isntall
#
# # set(SYCL_FLAGS "-fsycl" #       "-fsycl-targets=nvptx64-nvidia-cuda" #
# "-fsycl-unnamed-lambda" #       "-fsycl-device-code-split=off" #
# "-Wno-linker-warnings") set(SYCL_FLAGS "-fsycl"
# "-fsycl-targets=nvidia_gpu_sm_89" "-fsycl-unnamed-lambda"
# "-fsycl-device-code-split=off" "-fno-sycl-rdc"
# "-fsycl-device-code-split=per_kernel" "-Wno-linker-warnings")
#
# # Is always ignored # "--offload-arch=sm_89"
#
# # FROM DOCS: # Add the sources that require SYCL support to
# add_sycl_to_target(). # Not specifying any sources to add_sycl_to_target()
# adds SYCL compilation to all sources, # which may affect compilation time
# significantly:
#
# # SEtting build type to get dbugsymbols
#
# # TODO: ADD debugg symbols back in when you start using them... # if(NOT
# CMAKE_BUILD_TYPE) #     set(CMAKE_BUILD_TYPE Debug) # endif()
#
# # Compiler configuration # if(UNIX) #     # Direct CMake to use icpx rather
# than the default C++ compiler/linker #     set(CMAKE_CXX_COMPILER icpx) #
# else() # Windows #     # Force CMake to use icx-cl rather than the default C++
# compiler/linker #     # (needed on Windows only) # include(CMakeForceCompiler)
# #     CMAKE_FORCE_CXX_COMPILER(icx-cl IntelSYCL) # Use IntelSYCL instead of
# IntelDPCPP #     include(Platform/Windows-Clang) # endif()
#
# # To generate compile_commands.json -- useful for clangd LSP
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#
# # NOT SURE IF I WANT TO KEEP THIS FRom Popsift # Set build path as a folder
# named as the platform (linux, windows, darwin...) plus the processor type
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
# "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
# "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
#
# # TODO: Add more options like Popsift has if suitable
# option(PopSift_BUILD_EXAMPLES "Build PopSift applications."  ON) # JUst for a
# test option(BUILD_SHARED_LIBS "Build shared libraries" ON)
#
# # TODO: Seems to be beneficial for performance (keeping for now)
# message(STATUS "Building with -fPIC (Position Independent Code)")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#
# # TODO: Figure out if I should supprt visual studio (MSVC) -- Currently just
# doing -fPIC (shared libs  I guess)
#
# #
# ==============================================================================
# # GNUInstallDirs CMake module # - Define GNU standard installation directories
# # - Provides install directory variables as defined by the GNU Coding
# Standards. #
# ==============================================================================
# include(GNUInstallDirs)
#
# # Require threads because of std::thread. find_package(Threads REQUIRED)
#
# # TODO: Not sure what version this sould be set to and if it needs to be set
# at all
#
# # Set the required C++ standard for SYCL set(CMAKE_CXX_STANDARD 17) # Use 17
# or higher, depending on your SYCL implementation
# set(CMAKE_CXX_STANDARD_REQUIRED ON) # Enforce the standard
# set(CMAKE_CXX_EXTENSIONS OFF)       # Disable compiler-specific extensions
#
# add_subdirectory(src)
