# Followring documentation
if (CMAKE_HOST_WIN32)
  # need CMake 3.25.0+ for IntelLLVM support of target link properties on Windows
  cmake_minimum_required(VERSION 3.25)
  include (CMakeForceCompiler)
  CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP) # Should probably change this to IntelSYCL if possible
  include (Platform/Windows-Clang)
else()
  # CMake 3.23.5 is the minimum recommended for IntelLLVM on Linux
  cmake_minimum_required(VERSION 3.23.5)
  set(CMAKE_CXX_COMPILER icpx)
endif()

#
# if(UNIX)
#     # Direct CMake to use icpx rather than the default C++ compiler/linker
#     set(CMAKE_CXX_COMPILER icpx)
# else() # Windows
#     # Force CMake to use icx-cl rather than the default C++ compiler/linker 
#     # (needed on Windows only)
#     include (CMakeForceCompiler)
#     CMAKE_FORCE_CXX_COMPILER (icx-cl IntelDPCPP) # Should probably change this to IntelSYCL if possible
#     include (Platform/Windows-Clang)
# endif()


project(sycl_popSift VERSION 1.0.0 LANGUAGES CXX)

# Recommended over IntelDPCPP according to intel docs it might be deprecated in the future 
find_package(IntelSYCL REQUIRED) # should de added after project definition
# IntelSYCL -- should be  this but doeos not work need to fix isntall


# FROM DOCS:
# Add the sources that require SYCL support to add_sycl_to_target(). 
# Not specifying any sources to add_sycl_to_target() adds SYCL compilation to all sources, 
# which may affect compilation time significantly:




# Compiler configuration
# if(UNIX)
#     # Direct CMake to use icpx rather than the default C++ compiler/linker
#     set(CMAKE_CXX_COMPILER icpx)
# else() # Windows
#     # Force CMake to use icx-cl rather than the default C++ compiler/linker
#     # (needed on Windows only)
#     include(CMakeForceCompiler)
#     CMAKE_FORCE_CXX_COMPILER(icx-cl IntelSYCL) # Use IntelSYCL instead of IntelDPCPP
#     include(Platform/Windows-Clang)
# endif()


# To generate compile_commands.json -- useful for clangd LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# NOT SURE IF I WANT TO KEEP THIS FRom Popsift
# Set build path as a folder named as the platform (linux, windows, darwin...) plus the processor type
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")


# TODO: Add more options like Popsift has if suitable
option(PopSift_BUILD_EXAMPLES "Build PopSift applications."  ON) # JUst for a test
option(BUILD_SHARED_LIBS "Build shared libraries" ON)


# TODO: Seems to be beneficial for performance (keeping for now)
message(STATUS "Building with -fPIC (Position Independent Code)")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# TODO: Figure out if I should supprt visual studio (MSVC) -- Currently just doing -fPIC (shared libs  I guess)

# if(PopSift_USE_POSITION_INDEPENDENT_CODE AND NOT MSVC)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# endif()


# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# set(CMAKE_BUILD_TYPE Debug)
# if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
#   set(CMAKE_BUILD_TYPE Release)
#   message(STATUS "Build type not set, building in Release configuration")
#   # Set the possible values of build type for cmake-gui
#   set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
# else()
#   message(STATUS "Building in ${CMAKE_BUILD_TYPE} configuration")
# endif()

# ensure the proper linker flags when building the static version on MSVC
# if(MSVC AND NOT BUILD_SHARED_LIBS)
#   foreach(config "DEBUG" "RELEASE" "MINSIZEREL" "RELWITHDEBINFO")
#     string(REPLACE /MD /MT CMAKE_C_FLAGS_${config} "${CMAKE_C_FLAGS_${config}}")
#     string(REPLACE /MD /MT CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}}")
#     message(STATUS "CMAKE_C_FLAGS_${config} ${CMAKE_C_FLAGS_${config}}")
#     message(STATUS "CMAKE_CXX_FLAGS_${config} ${CMAKE_CXX_FLAGS_${config}}")
#   endforeach()
# endif()

# ==============================================================================
# GNUInstallDirs CMake module
# - Define GNU standard installation directories
# - Provides install directory variables as defined by the GNU Coding Standards.
# ==============================================================================
include(GNUInstallDirs)

# Require threads because of std::thread.
find_package(Threads REQUIRED)

# TODO: Not sure what version this sould be set to and if it needs to be set at all

# Set the required C++ standard for SYCL
set(CMAKE_CXX_STANDARD 17) # Use 17 or higher, depending on your SYCL implementation
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Enforce the standard
set(CMAKE_CXX_EXTENSIONS OFF)       # Disable compiler-specific extensions


add_subdirectory(src)


# TODO: Add to following commented out code to the applicatin cmakelists.txt

# Only needed for application not the library
# # Find Boost libraries
# find_package(Boost 1.70 REQUIRED COMPONENTS filesystem program_options)
#
# # Look for DevIL library
# find_package(DevIL COMPONENTS IL ILU) # yields IL_FOUND, IL_LIBRARIES, IL_INCLUDE_DIR

# This is a Windows-specific flag that enables exception handling in host code
# if(WIN32)
#     set(WIN_FLAG "/EHsc")
# endif()

# Lists that we build and include in the target
# set(PD_INCLUDE_DIRS    ${Boost_INCLUDE_DIRS})
# set(PD_LINK_LIBS       ${Boost_LIBRARIES}) 


# This belongs in applicaton's CMakeLists
# if(IL_FOUND OR DevIL_FOUND)
#   message(STATUS "DevIL found, setting USE_DEVIL macro")
#   set(PD_COMPILE_OPTIONS "-DUSE_DEVIL") # sets macro USE_DEVIL so we can use #ifdef in code
#
#   # Add DevIL libraries to the target lists
#   list(APPEND PD_INCLUDE_DIRS ${IL_INCLUDE_DIR})
#   list(APPEND PD_LINK_LIBS    ${IL_LIBRARIES} ${ILU_LIBRARIES})
# else()
#   message(WARNING "DevIL not found -- Falling back to pgmread")
#   set(PD_COMPILE_OPTIONS "" )
# endif()

# TODO: ADD support for windows with MSVC like in popsift


# set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG} ${PD_COMPILE_OPTIONS}")
# set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG}")
# set(LINK_FLAGS "-fsycl")


# add_executable(${TARGET_NAME} ${SOURCE_FILE})
# set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
# set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
#
# target_include_directories(${TARGET_NAME} PUBLIC ${PD_INCLUDE_DIRS})
# target_link_libraries(${TARGET_NAME} ${PD_LINK_LIBS})
