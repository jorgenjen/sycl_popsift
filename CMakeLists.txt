cmake_minimum_required(VERSION 3.12)
project(sycl_popsift)

# if(CMAKE_BUILD_TYPE STREQUAL "") message(STATUS "CMAKE_BUILD_TYPE was not set,
# setting to Release") set(CMAKE_BUILD_TYPE "Release") endif()

set(CMAKE_CXX_STANDARD 17) # required for sycl

# # Require threads because of std::thread. find_package(Threads REQUIRED)

# TODO: Add more options like Popsift has if suitable
option(PopSift_BUILD_EXAMPLES "Build PopSift applications." ON)

option(PopSift_EXPERIMENTAL "Use experimantal features." ON) # bindless images
# option(BUILD_SHARED_LIBS "Build shared libraries" ON)
# These are toggles that only have effect whem PopSift_EXPERIMENTAl are ON
option(BindlessInput "Use bindless input image for upscaling" ON) 
option(BindlessArray "Use bindless image array for pyramid" ON)
#
# # TODO: Seems to be beneficial for performance (keeping for now)
# message(STATUS "Building with -fPIC (Position Independent Code)")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

# TODO: Make it like this for amd and intel gpu's too so you can toggle on and
# off cpu only
if(${ENABLE_CUDA})
  message(STATUS "CUDA enabled")
  # set(CPU_ONLY "")
else()
  message(STATUS "CUDA DISABLED -- Running cpu only mode")
  # set(CPU_ONLY "-DCPU_ONLY")

  add_compile_definitions(CPU_ONLY)

  # set(CMAKE_BUILD_TYPE Debug) message(STATUS "Running in debug mode :D")
  # list(APPEND SYCL_FLAGS -O0 -g -fno-omit-frame-pointer)
  set(CMAKE_BUILD_TYPE Release)
  list(APPEND SYCL_FLAGS -O3)
endif()

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
else()
# Non CUDA
if(PopSift_EXPERIMENTAL)
    message(STATUS "Using experimantal features!")
    # Enable none cuda exlusive extensions
else()
    message(STATUS "Building without experimental features")
    # Disable none cuda exclusive extensions
endif()

  # set(EXPERIMENTAL "-DUSE_EXPERIMENTAL 0") set(EXPERIMENTAL "")

  # Currentky bindless images are cuda exlusive
  add_compile_definitions(USE_BINDLESS_INPUT=0)
  add_compile_definitions(USE_BINDLESS_ARRAY=0)
  
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
